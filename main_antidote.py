
import os
import numpy
import argparse
from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import pprint
from utils import *
from config import *
from tqdm import tqdm
from datasets.data_loaders import data_loader
import scipy.optimize as opt

parser = argparse.ArgumentParser(description='Robust Loss Functions for Learning with Noisy Labels: Benchmark Datasets')
# dataset settings
parser.add_argument('--dataset', type=str, default="cifar100", choices=['mnist', 'cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--root', type=str, default="../database", help='the dataset root, change root yourself')
parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric'], 
                    help='label noise type. use clean label by setting noise rate = 0')
parser.add_argument('--noise_rate', type=str, default='0.2', 
                    help='the noise rate 0~1. if using human noise, should set in [clean, worst, aggre, rand1, rand2, rand3, clean100, noisy100]')
parser.add_argument('--noise_method', type=str, default='method1', choices=['method1, method2'], 
                    help='different code implementation for symmetric and asymmetric noise, will cause little performance differences'
                         'this does not affect dependent and human noise')
# initialization settings
parser.add_argument('--gpus', type=str, default='0', help='the used gpu id')
parser.add_argument('--seed', type=int, default=123, help='initial seed')
parser.add_argument('--trials', type=int, default=3, help='number of trials')
parser.add_argument('--test_freq', type=int, default=1, help='epoch frequency to evaluate the test set')
parser.add_argument('--save_model', default=False, action="store_true", help='whether to save trained model')
# training settings
# loss: antidote_kl_bisec, ECEandMAE: Eps-Softmax with CE loss (ECE) and MAE; EFLandMAE: Eps-Softmax with FL loss (EFL) and MAE
parser.add_argument('--loss', type=str, default='antidote_kl_bisec', help='the loss function: CE, ECEandMAE, EFLandMAE, GCE ... ')

# args specific to ANTIDOTE
parser.add_argument('--lr_decay_milestones', default=[100], help='the epoch to apply lr scheduler')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='rate of decay for the step-wise scheduler for both model and antidote parameters')
parser.add_argument('--lr_', type=float, default=0.001, help='the epoch to apply lr scheduler')
parser.add_argument('--lambda_', type=float, default=0.1, help='the epoch to apply lr scheduler')
parser.add_argument('--alpha_', type=float, default=2., help='the degree of alpha divergence')
parser.add_argument('--rho_', type=float, default=-5., help='rho parameter for antidote alpha')
parser.add_argument('--delta_', type=float, default=0.27, help='the epoch to apply lr scheduler')
parser.add_argument('--lambda_failsafe', type=float, default=0.01, help='value for lambda enforced every time lambda goes negative')
parser.add_argument('--t_', type=float, default=0., help='ratio for interpolation with CE')
parser.add_argument('--t_step', default=0, help='number of initial epochs heavily interpolated with CE (warmup)')
parser.add_argument('--k_', type=float, default=0.05, help='lower bound for lambda incorporated in the loss as a regularizer')
parser.add_argument("--restarts", default=0, help='number of times to restart if accuracy threshold is not met after warmup')
parser.add_argument('--restart_epoch', default=20, help='epoch number to check for restart')
parser.add_argument('--restart_th', type=float, default=0.14, help='minimum accuracy that warrants a restart if no achieved at restart_epoch')
parser.add_argument('--scheduler', default='step', help='step vs. cosine scheduler for learning rate of the model and antidote parameters')
parser.add_argument("--verbose", action="store_true", help="prints antidote parameters")

args = parser.parse_args()
args.dataset = args.dataset.lower()

# change root yourself
if args.dataset == 'cifar10': 
    args.root = args.root + '/CIFAR10'
elif args.dataset == 'cifar100':
    args.root = args.root + '/CIFAR100'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
if torch.cuda.is_available():
    device = 'cuda'
    # canceling this can completely fix random seed, but will slow down your training
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'
print('We are using', device)

loss_fct = nn.CrossEntropyLoss(reduce=False)

def f_star_prime(t,alpha):
    return numpy.maximum((alpha-1.)*t.cpu().detach().numpy(),0.)**(1./(alpha-1.))

def rho_objective(rho_,lambda_,k, loss_array,alpha):
    return -1.+numpy.average(f_star_prime(-(loss_array.cpu().detach().numpy()+rho_)/(lambda_.detach()+k),alpha))

def optimize_rho(lambda_,k, loss_array,alpha,a=-1,b=1):
    #a,b are starting guesses for interval endpoints (assumes a<b) over which zero occurs
    opposite_sign=False
    while not opposite_sign:
        obj_a=rho_objective(a,lambda_,k,loss_array,alpha)
        obj_b=rho_objective(b,lambda_,k,loss_array,alpha)
        if numpy.sign(obj_a)==numpy.sign(obj_b):
            Delta=b-a
            a=a-Delta
            b=b+Delta
        else:
            opposite_sign=True

    rho_star=opt.bisect(lambda rho_: rho_objective(rho_,lambda_.cpu().detach(),k,loss_array,alpha),a,b)
    
    return rho_star, a, b

def f_ (t, alpha_):
    return (t**alpha_ - 1) / (alpha_ * (alpha_ - 1))

def lambda_objective_derivative(delta,k_,lambda_,loss_array):
    
    min_loss=numpy.min(loss_array)
    return -delta-numpy.log(numpy.average(numpy.exp((-loss_array+min_loss)/(lambda_+k_))))+\
        min_loss/(lambda_+k_)-(lambda_+k_)**(-1)*numpy.average(loss_array*numpy.exp((-loss_array+min_loss)/(lambda_+k_)))/numpy.average(numpy.exp((-loss_array+min_loss)/(lambda_+k_)))


def optimize_lambda(delta,k_,loss_array,lambda_b=10,lambda_max=100):
    if lambda_objective_derivative(delta,k_,0,loss_array)<=0:
        return 0
    else:
        n=0
        while lambda_objective_derivative(delta,k_,lambda_b,loss_array)>=0:
            lambda_b=2*lambda_b
            if lambda_b>=lambda_max:
                return lambda_max
    
        lambda_star=opt.bisect(lambda z: lambda_objective_derivative(delta,k_,z,loss_array),0,lambda_b)

        return lambda_star


def train(args, i):
    seed_everything(args.seed + i)
    # The code base supports the simple mnist dataset for symmetric and asymmetric noise, but we do not use it in the paper
    if args.dataset == 'mnist': 
        epochs = 50
        lr = 0.01
        batch_size = 128
        model = CNN(type='mnist').to(device)
    elif args.dataset == 'cifar10':
        epochs = 120
        lr = 0.05
        batch_size = 128
        model = CNN(type='cifar10').to(device)
    elif args.dataset == 'cifar100':
        epochs = 200
        lr = 0.05
        batch_size = 128
        model = ResNet34(num_classes=100).to(device)
    else:
        raise NotImplementedError

    logger.info('\n' + pprint.pformat(args))
    l1_weight_decay, l2_weight_decay = get_weight_decay_config(args)
    logger.info('lr={}, batch_size={}, l1_weight_decay={}, l2_weight_decay={}'.format(lr, batch_size, l1_weight_decay, l2_weight_decay))
    
    train_loader, test_loader = data_loader(args=args, batch_size=batch_size)

    criterion = get_loss_config(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_weight_decay)
    
    ## ANTIDOTE training
    if (args.loss in ['antidote_alpha', 'antidote_kl', 'antidote_kl_bisec']):
        if not args.loss=='antidote_kl_bisec':
            optimizer_lambda = torch.optim.AdamW(list(criterion.parameters()), lr=criterion.lr_)
        if args.scheduler=='step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_milestones, gamma=args.lr_decay_rate)
            if not args.loss=='antidote_kl_bisec':
                sched_lambda = torch.optim.lr_scheduler.MultiStepLR(optimizer_lambda, milestones=args.lr_decay_milestones, gamma=args.lr_decay_rate)
        elif args.scheduler =='cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=0.0)
            if not args.loss=='antidote_kl_bisec':
                sched_lambda = torch.optim.lr_scheduler.StepLR(optimizer_lambda, step_size=args.lr_decay_milestones[0], gamma=args.lr_decay_rate)
        
        for epoch in tqdm(range(epochs), ncols=60, desc=args.loss + ' ' + args.dataset):
            model.train()
            total_loss = 0.
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                if not args.loss=='antidote_kl_bisec':
                    optimizer_lambda.zero_grad()              
                
                out = model(batch_x)
                antidote_loss = criterion(out, batch_y)
                loss_CE = loss_fct(out, batch_y).mean()
                if epoch >= args.t_step:
                    loss = (1-args.t_) * antidote_loss + args.t_ * loss_CE
                else:
                    loss = args.t_ * antidote_loss + (1-args.t_) * loss_CE
           
                if args.loss =='antidote_alpha':
                    with torch.no_grad():
                        ## update rho via bisection method
                        l = nn.CrossEntropyLoss(reduction='none')(out,batch_y)
                        rho_star,a,b = optimize_rho(criterion.lambda_, criterion.k_, l, criterion.alpha_, criterion.a_, criterion.b_)
                        criterion.rho = rho_star
                
                if l1_weight_decay != 0:
                    l1_decay = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_weight_decay * l1_decay
                
                loss.backward()
                # gradient norm bound, following previous work setting
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                
                # gradient reversal of lambda for maximization
                if not args.loss=='antidote_kl_bisec':
                    criterion.lambda_.grad.data = -criterion.c_ * criterion.lambda_.grad.data
                else:
                     with torch.no_grad():
                        ## update lambda via bisection method
                        l = nn.CrossEntropyLoss(reduction='none')(out,batch_y)
                        lambda_star = optimize_lambda(args.delta_, args.k_, l.cpu().detach().numpy())
                        criterion.lambda_ = torch.tensor(lambda_star)
                
                optimizer.step()
                if not args.loss=='antidote_kl_bisec':
                    optimizer_lambda.step()
                if (criterion.lambda_<0.):
                    with torch.no_grad():
                        criterion.lambda_.copy_(torch.tensor(args.lambda_failsafe))
                        print("fail-safe activated.")
                total_loss += loss.item()
            scheduler.step()
            if not args.loss=='antidote_kl_bisec':
                sched_lambda.step()
            if (epoch + 1) % args.test_freq == 0:
                test_acc, _ = evaluate(test_loader, model, device)
                logger.info('Iter {}: loss={:.4f}, lambda={:.4f}, test_acc={:.4f}'.format(epoch, total_loss, criterion.lambda_.item(), test_acc))
            if args.restarts>0:
                if epoch+1 ==args.restart_epoch and test_acc < args.restart_th:
                    break
            if (args.verbose):
                if args.loss == 'antidote_alpha':
                    print('rho= ',rho_star, 'a= ', a, 'b= ',b, 'lambda= ',criterion.lambda_.item(), 'loss= ',loss.item())
                elif args.loss == 'antidote_kl':
                    print('lambda= ',criterion.lambda_.item(), 'loss= ',loss.item())
        if args.save_model:
            torch.save(model, results_path + '/model.pth')
        # return last epoch test acc
        return test_acc
    ## traditional training
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        for epoch in tqdm(range(epochs), ncols=60, desc=args.loss + ' ' + args.dataset):
            model.train()
            total_loss = 0.
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                if l1_weight_decay != 0:
                    l1_decay = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_weight_decay * l1_decay
                loss.backward()
                # gradient norm bound, following previous work setting
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) 
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % args.test_freq == 0:
                test_acc, _ = evaluate(test_loader, model, device)
                logger.info('Iter {}: loss={:.4f}, test_acc={:.4f}'.format(epoch, total_loss, test_acc))
        if args.save_model:
            torch.save(model, results_path + '/model.pth')
        # return last epoch test acc
        return test_acc 
    
if __name__ == "__main__":
    tag = f"default"
    results_path = os.path.join('./results', args.dataset, args.loss, args.noise_type + '_' + args.noise_rate, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = get_logger(results_path + '/result.log')
    accs = []
    trials = args.trials+args.restarts  
    for i in range(trials):    
        acc = train(args, i)
        if args.restarts==0:
            accs.append(acc)
        elif acc>=args.restart_th:
            accs.append(acc)
    accs = torch.asarray(accs)*100
    logger.info(args.dataset+' '+args.loss+': %.2f±%.2f \n' % (accs.mean(), accs.std()))


    
