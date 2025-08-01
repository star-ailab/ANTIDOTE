
import os
import numpy
import argparse
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import *
from config import *
from lightning import Fabric
from datasets.data_loaders import webvision_loader
import pprint
import scipy.optimize as opt

parser = argparse.ArgumentParser(description='Robust Loss Functions for Learning with Noisy Labels: Real-World Datasets')
# dataset settings
parser.add_argument('--dataset', type=str, default="webvision", choices=['webvision'], help='dataset name')
parser.add_argument('--root', type=str, default="../database", help='the data root')
# initialization settings
parser.add_argument('--gpus', type=str, default='0', help='0 or 1, per id corresponding 4 gpus, change by yourself')
parser.add_argument('--grad_bound', type=bool, default=True, help='the gradient norm bound, following previous work')
parser.add_argument('--loss', type=str, default='antidote_kl_bisec', help='the loss function: antidote_kl_bisec, ECEandMAE, EFLandMAE ... ')

# args specific to ANTIDOTE
parser.add_argument('--lr_decay_milestones', default=[124,199], help='the epoch to apply lr scheduler')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='rate of decay for the step-wise scheduler for both model and antidote parameters')
parser.add_argument('--lr_', type=float, default=0.1, help='the epoch to apply lr scheduler')
parser.add_argument('--lambda_', type=float, default=1., help='the epoch to apply lr scheduler')
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

# change gpu id yourself
if args.gpus == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
elif args.gpus == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
gpu_nums = torch.cuda.device_count() 
# using bf16-mixed precision for training
fabric = Fabric(accelerator='cuda', devices='auto', strategy='ddp', precision='bf16-mixed')
fabric.launch()

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


if args.dataset == 'webvision':
    lr = 0.4
    epochs = 250
    l1_weight_decay, l2_weight_decay = get_weight_decay_config(args)
    batch_size = int(256 / gpu_nums)
    # change root yourself
    args.root = args.root + '/WebVision'
    args.grad_bound = True
    nesterov = True
    model = resnet50(num_classes=50, zero_init_residual=True)
    train_loader, test_loader, img_test_loader = webvision_loader(args, batch_size)


if torch.distributed.get_rank() == 0:
    tag = f"default"
    results_path = os.path.join('./results', args.dataset, args.loss, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = get_logger(results_path + '/result.log')

    logger.info('\n' + pprint.pformat(args))
    logger.info('lr={}, batch_size={}, l1_weight_decay={}, l2_weight_decay={}'.format(lr, batch_size, l1_weight_decay, l2_weight_decay))

criterion = get_loss_config(args)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=nesterov, weight_decay=l2_weight_decay)
model, optimizer = fabric.setup(model, optimizer)
if args.dataset == 'webvision':
    scheduler = StepLR(optimizer, step_size=5, gamma=0.97)
    train_loader, test_loader, img_test_loader= fabric.setup_dataloaders(train_loader, test_loader, img_test_loader)

if torch.distributed.get_rank() == 0:
    epochs_iterator = tqdm(range(epochs), ncols=60, desc=args.loss + ' ' + args.dataset)
else:
    epochs_iterator = range(epochs)

## ANTIDOTE training
if (args.loss in ['antidote_alpha', 'antidote_kl', 'antidote_kl_bisec']):
    if not args.loss=='antidote_kl_bisec':
        optimizer_lambda = torch.optim.AdamW(list(criterion.parameters()), lr=criterion.lr_)
    if args.scheduler=='step_const':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_milestones, gamma=args.lr_decay_rate)
        if not args.loss=='antidote_kl_bisec':
            sched_lambda = StepLR(optimizer_lambda, step_size=1, gamma=0.97)
    elif args.scheduler=='step':
        if not args.loss=='antidote_kl_bisec':
            sched_lambda = StepLR(optimizer_lambda, step_size=1, gamma=0.97)
    elif args.scheduler =='cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=0.0)
        if not args.loss=='antidote_kl_bisec':
            sched_lambda = torch.optim.lr_scheduler.StepLR(optimizer_lambda, step_size=args.lr_decay_milestones[0], gamma=args.lr_decay_rate)
    
    for epoch in epochs_iterator:
            model.train()
            total_loss = 0.
            for batch_x, batch_y in train_loader:
                model.zero_grad()
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
                
                fabric.backward(loss)
                if args.grad_bound:
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
            test_acc1, test_acc5 = evaluate(test_loader, model)
            test_acc1, test_acc5 = fabric.all_gather(test_acc1).mean(), fabric.all_gather(test_acc5).mean()
            
            ##
            if (args.verbose):
                if args.loss == 'antidote_alpha':
                    print('rho= ',rho_star, 'a= ', a, 'b= ',b, 'lambda= ',criterion.lambda_.item(), 'loss= ',loss.item())
                elif args.loss == 'antidote_kl':
                    print('lambda= ',criterion.lambda_.item(), 'loss= ',loss.item())
            
            if args.dataset == 'webvision':
                img_acc1, img_acc5 = evaluate(img_test_loader, model)
                img_acc1, img_acc5 = fabric.all_gather(img_acc1).mean(), fabric.all_gather(img_acc5).mean()
        
            if torch.distributed.get_rank() == 0:
                if args.dataset == 'webvision':
                    logger.info('Iter {}: loss={:.2f}, antidote_lamda={:.2f}, web_acc1={:.4f}, web_acc5={:.4f}, img_acc1={:.4f}, img_acc5={:.4f}'.format(epoch, total_loss, criterion.lambda_.item(), test_acc1, test_acc5, img_acc1, img_acc5))
                else:
                    logger.info('Iter {}: loss={:.2f}, antidote_lamda={:.2f}, test_acc1={:.4f}, test_acc5={:.4f}'.format(epoch, total_loss, criterion.lambda_.item(), test_acc1, test_acc5))

## traditional training
else:
    for epoch in epochs_iterator:
        model.train()
        total_loss = 0.
        for batch_x, batch_y in train_loader:
            model.zero_grad()
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            if l1_weight_decay != 0:
                l1_decay = sum(p.abs().sum() for p in model.parameters())
                loss += l1_weight_decay * l1_decay
            fabric.backward(loss)
            if args.grad_bound:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += loss.item()
    
        scheduler.step()
        test_acc1, test_acc5 = evaluate(test_loader, model)
        test_acc1, test_acc5 = fabric.all_gather(test_acc1).mean(), fabric.all_gather(test_acc5).mean()
        if args.dataset == 'webvision':
            img_acc1, img_acc5 = evaluate(img_test_loader, model)
            img_acc1, img_acc5 = fabric.all_gather(img_acc1).mean(), fabric.all_gather(img_acc5).mean()
    
        if torch.distributed.get_rank() == 0:
            if args.dataset == 'webvision':
                logger.info('Iter {}: loss={:.2f}, web_acc1={:.4f}, web_acc5={:.4f}, img_acc1={:.4f}, img_acc5={:.4f}'.format(epoch, total_loss, test_acc1, test_acc5, img_acc1, img_acc5))
            else:
                logger.info('Iter {}: loss={:.2f}, test_acc1={:.4f}, test_acc5={:.4f}'.format(epoch, total_loss, test_acc1, test_acc5))

