import  torch, os
import  numpy as np
from    SeismicNShot import SeismicNShot
import  argparse
import torch.nn as nn
from    model import METANAFNet
from collections import OrderedDict
from msssimLoss import MSSSIM
from torch.utils.tensorboard import SummaryWriter

'''
Define the dataset for meta-train and meta-test.
If you download the dataset, you can modify here accordingly.
'''
dir_interpolation = '../meta_train_dataset/interpolation/'      
dir_random_denoise = '../meta_train_dataset/denoise/'           
dir_groundroll_denoise = '../meta_train_dataset/ground_roll/'   
dir_migration = '../meta_train_dataset/migration/'
dir_vrms = '../meta_train_dataset/vrms/'

def main(args):

    # set seed
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # print your args
    print(args)

    # define device
    device = torch.device('cuda')

    # define Meta network
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    width = 32

    maml = METANAFNet(in_channels=1, width=width, middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    # define Meta optimizer, which is activate in outer loop
    meta_optimizer = torch.optim.AdamW(maml.parameters(), lr=args.meta_lr)

    # define loss function, which include MSE and MSSSIM
    criterion = nn.MSELoss()
    criterion2 = MSSSIM()

    # define learning rate variation for outer loop
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=2000, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5000, 15000], gamma=0.4)

    # Meta training dataset load
    db_train = SeismicNShot(batchsz=args.task_num, k_shot=args.k_spt, k_query=args.k_qry, imgsz=args.imgsz, 
                dir_interpolation = dir_interpolation, dir_random_denoise = dir_random_denoise, 
                dir_groundroll_denoise = dir_groundroll_denoise, dir_migration = dir_migration, dir_vrms = dir_vrms)

    # log file
    writer = SummaryWriter(comment=f'MetaLR_{args.meta_lr}_UpdateLR_{args.update_lr}_Epoch_{args.epoch}_ \
                        Updatestep_{args.update_step}')

    # meta model file
    try:
        os.makedirs('./checkpoints')
    except OSError:
        pass

    # inner loss list, where args.task_num reprsent how many seismic processing task
    inner_loss_list=np.zeros([args.task_num,1])

    for step in range(args.epoch):
        # iter support data set and query data set
        input_spt, label_spt, input_qry, label_qry = db_train.next()
        input_spt, label_spt, input_qry, label_qry = torch.from_numpy(input_spt).to(device), torch.from_numpy(label_spt).to(device), \
                                     torch.from_numpy(input_qry).to(device), torch.from_numpy(label_qry).to(device)

        maml.zero_grad()

        # initial outer loop loss and msssim
        outer_loss = torch.tensor(0., device=device)
        ms_ssim = torch.tensor(0., device=device)

        # define learning rate variation for inner loop
        if step !=0 and step % 2000 ==0 and step>=2000 and step<20000:
            args.update_lr = args.update_lr*0.8

        # Update parameters on the support set for each task, and use the updated parameters to estimate on the query set
        for i in range(args.task_num):

            params = OrderedDict(maml.named_parameters())

            for k in range(args.update_step):

                pred = maml.functional_forward(input_spt[i], params=params)

                inner_loss = args.scale*(criterion(pred, label_spt[i]) + 1 - criterion2(pred, label_spt[i]))

                maml.zero_grad()

                grads = torch.autograd.grad(inner_loss, params.values(), create_graph=not args.first_order)

                params = OrderedDict(
                        (name, param - args.update_lr * grad)
                        for ((name, param), grad) in zip(params.items(), grads))

            if (step + 1) % 100 == 0:
                inner_loss_list[i]=inner_loss.item()

            pred_qry = maml.functional_forward(input_qry[i], params=params)

            outer_loss +=criterion(pred_qry, label_qry[i])+1-criterion2(pred_qry, label_qry[i])

            with torch.no_grad():
                ms_ssim += criterion2(pred_qry, label_qry[i])

        outer_loss = args.scale * outer_loss / args.task_num

        meta_optimizer.zero_grad()

        outer_loss.backward()

        meta_optimizer.step()

        writer.add_scalar('Loss/meta_loss', outer_loss.item(), step)
        writer.add_scalar('Loss/inner_loss', inner_loss.item(), step)
        writer.add_scalar('Accs/ms-ssim', ms_ssim.item(), step)

        if (step + 1) % 100 == 0:
            print('step:', step + 1, '\Inner loss:', inner_loss.item())
            print('step:', step + 1, '\Outer loss:', outer_loss.item())
            print('step:', step + 1, '\MS-SSIM:', (ms_ssim / args.task_num).item())
            print('step:', step + 1, '\All task:', inner_loss_list)

        if (step + 1) % 100 == 0:
            torch.save(maml.state_dict(), './checkpoints/'+f'CP_epoch{step + 1}.pth')

        scheduler.step()

    writer.close()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=8)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=12)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--scale', type=float, help='scaling factor adjust the loss value', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--first_order', type=str, help='whether first order approximation of MAML is used', default=True)


    args = argparser.parse_args()

    main(args)
