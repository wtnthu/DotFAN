import datetime
import os
import time
from torchvision.utils import save_image
from recognition_pose import mobilenet_v1
import model
from Loss import*
import tensorflow as tf
from random import shuffle
from matplotlib import pyplot as plt
import random
def plot_tensor(x):
    out = (x + 1) / 2
    show = out.cpu().numpy()[0].transpose( (1,2,0)).astype('float32')
    if show.shape[2]==1:
        plt.imshow(show[:,:,0])
    else:
        plt.imshow(show[:, :])
    plt.show()
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)


        self.writer.add_summary(summary, step)
class Solver(object):

    def __init__(self, train_loader, train_loaderB, test_loader, config):
        # Data loader

        self.train_loader = train_loader
        self.train_loaderB = train_loaderB
        self.test_loader = test_loader
        self.config = config
        self.logger_dict = {}
        self.rec_loss = L1_Charbonnier_loss()
        self.l2_criterion = L1_Charbonnier_loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

        # Start with trained model
        if self.config.pretrained_model:
            self.load_pretrained_model(self.config.pretrained_model)

    def build_model(self):
        if self.config.gray:
            self.in_channel=1
        else:
            self.in_channel=3

        checkpoint_fp = 'data/pretrained_ckpt/'+self.config.P_pretrained
        arch = 'mobilenet_1'
        self.D3MM_model = getattr(mobilenet_v1, arch)(num_classes=240,input_channel=self.in_channel)  # 62 = 12(pose) + 40(shape) +10(expression)
        checkpoint = torch.load(checkpoint_fp)['state_dict']
        self.D3MM_model.load_state_dict(checkpoint)
        self.D3MM_model = self.D3MM_model.to(self.device)
        self.D3MM_model.eval()

        from recognition_pose.insightface import Backbone
        self.R = Backbone(50, 0.6, 'ir_se',s=64 ).to(self.device)
        self.R.load_state_dict(torch.load('data/pretrained_ckpt/'+self.config.L_pretrained))
        self.R.eval()

        Generator = getattr(model, self.config.G_net)
        Discriminator = getattr(model, self.config.D_net)

        if self.config.norm== "instance":
            Norm = nn.InstanceNorm2d
            print('instance')
        elif self.config.norm== "batch":
            Norm = nn.BatchNorm2d
            print('batch')

        c_dim = 512 +  self.config.c_dim

        self.G = Generator(self.config.g_conv_dim, c_dim+self.config.c2_dim+1, self.config.g_repeat_num,self.in_channel,Norm=Norm)
        self.D = Discriminator(self.config.face_crop_size, self.config.d_conv_dim, self.config.c2_dim,self.config.d_repeat_num,self.in_channel)


        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        # Print networks

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def load_pretrained_model(self,epoch):
        self.G.load_state_dict(torch.load(os.path.join(
            self.config.model_save_path, '{}_G.pth'.format(epoch))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.config.model_save_path, '{}_D.pth'.format(epoch))))
        print('loaded trained models (step: {})..!'.format(epoch))
    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    def to_var(self, x, volatile=True):
        return x.to(self.device)
    def h_flip(self,tensor,dim=1):
        inv_idx = self.to_var(torch.range(tensor.size(dim)-1,0, -1).long())
        return tensor.index_select(dim, inv_idx)
    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    def save_model(self,e):
        torch.save(self.G.state_dict(),
                   os.path.join(self.config.model_save_path, '{}_G.pth'.format(e + 1)))

        torch.save(self.D.state_dict(),
                    os.path.join(self.config.model_save_path, '{}_D.pth'.format(e + 1)))

    def save_example(self,fixed_x,pose_c_list,illumination_c_list,e,i):
        self.G.eval()
        fake_image_list = [fixed_x]
        shuffle(illumination_c_list)
        for fixed_c_index in range(len(pose_c_list)):
            attribute_pose = pose_c_list[fixed_c_index].unsqueeze(0)
            attribute_illumination = illumination_c_list[fixed_c_index].unsqueeze(0)
            attribute = torch.cat([attribute_pose,attribute_illumination], dim=1)
            attribute = attribute.expand(fixed_x.size(0),attribute.size(1))
            fake_image_list.append(self.G_operation(fixed_x,attribute)[0])
        fake_images = torch.cat(fake_image_list, dim=3)
        save_image(self.denorm(fake_images.data),
                   os.path.join(self.config.sample_path, '{}_{}_fake.png'.format(e + 1,i)), nrow=1, padding=0)
        print('Translated images and saved into {}..!'.format(self.config.sample_path))
        self.G.train()
    def G_operation(self,x,c):
        identity_codes = []
        identity_code = self.R(x, 'test',self.config.gray)
        identity_codes.append(identity_code)
        fake_x, _ = self.G(x, c,identity_codes)
        return fake_x,identity_codes

    def make_celeb_labels(self,c_dim=14):
        fixed_c_list = []
        for i in range(c_dim):
            fixed_c = np.zeros([c_dim])
            for k in range(c_dim):
                fixed_c[k] = 0
            fixed_c[i] = 1
            fixed_c_list.append(self.to_var(torch.FloatTensor(fixed_c), volatile=True))

        return fixed_c_list
    def change_pose_without_shape(self,real_pose,pointed_pose):
        real_shape = real_pose[:, 12:211]
        target_pose = Variable(torch.cat((pointed_pose[:, 0:12], real_shape, pointed_pose[:, 211:]), 1).data)
        return target_pose
    def train(self):
        self.display_counter = 0
        g_lr = self.config.g_lr
        d_lr = self.config.d_lr

        if self.config.pretrained_model:
            start = int(self.config.pretrained_model.split('_')[0])-1
            for e in range(0,start):
                if (e+1) > (self.config.num_epochs - self.config.num_epochs_decay):
                    g_lr -= (self.config.g_lr / float(self.config.num_epochs_decay))
                    d_lr -= (self.config.d_lr / float(self.config.num_epochs_decay))
                    self.update_lr(g_lr, d_lr)
        else:
            start = 0

            # Set dataloader

        self.data_loader_A = self.train_loader
        data_iter_A = iter(self.data_loader_A)

        self.data_loader_B = self.train_loaderB
        data_iter_B = iter(self.data_loader_B)

        iters_per_epoch = len(self.data_loader_A)

        fixed_x_A, real_illumination_A,_  =  next(data_iter_A)
        fixed_x_B, real_illumination_B, _ =  next(data_iter_B)

        fixed_x = torch.cat([fixed_x_A[:3],fixed_x_B[:3]], dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        fixed_x_idx = torch.randperm(fixed_x.size(0))
        real_x_random = fixed_x[fixed_x_idx]

        fixed_illumination_list = self.make_celeb_labels()
        real_pose_x_ = Variable((self.D3MM_model(fixed_x)[:, 0:self.config.c_dim]).data)

        if self.config.c_dim > 12:
            pointed_pose = self.D3MM_model(real_x_random)
            fixed_pose_list = self.change_pose_without_shape(real_pose_x_,pointed_pose)
        else:
            fixed_pose_list = Variable(self.D3MM_model(real_x_random)[:, 0:self.config.c_dim].data)
        self.loss = {}

        start_time = time.time()

        for e in range(start, self.config.num_epochs):

            for i in range(0, iters_per_epoch):
                if (i) % (iters_per_epoch//50) == 0:
                    self.save_example(fixed_x, fixed_pose_list,fixed_illumination_list, e,i)
                    self.val(e)


                try:
                    real_x_A, real_illumination_A,mask_A = next(data_iter_A)
                except:
                    data_iter_A = iter(self.data_loader_A)
                    real_x_A, real_illumination_A, mask_A = next(data_iter_A)

                try:
                    real_x_B, real_illumination_B, mask_B = next(data_iter_B)
                except:
                    data_iter_B = iter(self.data_loader_B)
                    real_x_B, real_illumination_B, mask_B = next(data_iter_B)

                # Convert tensor to variable
                real_x =  torch.cat([real_x_A,real_x_B], dim=0)
                real_x = self.to_var(real_x)
                mask_x = torch.cat([mask_A, mask_B], dim=0)
                mask_x = self.to_var(mask_x)
                if i % 2 == 0:
                    illumination_list =  torch.cat([real_illumination_B,real_illumination_B], dim=0) # use pie illumination for cls
                    illumination_list = self.to_var(illumination_list)
                    rand_idx = torch.randperm(real_x.size(0))
                    fake_illumination = illumination_list[rand_idx]
                    real_label = illumination_list.clone()[:,1:]
                    fake_label = fake_illumination.clone()[:,1:]

                else:
                    illumination_list = torch.cat([real_illumination_A, real_illumination_B], dim=0)  # use original illumination for rec
                    illumination_list = self.to_var(illumination_list)
                    rand_idx = torch.randperm(real_x.size(0)//2)
                    fake_illumination = torch.cat([real_illumination_A,real_illumination_B[rand_idx]], dim=0)
                    fake_illumination = self.to_var(fake_illumination)
                    real_label = illumination_list.clone()[:,1:]
                    fake_label = fake_illumination.clone()[:,1:]


                rand_idx = torch.randperm(real_x.size(0))
                real_x_random = real_x[rand_idx]



                real_pose = Variable((self.D3MM_model(real_x)[:,0:self.config.c_dim]).data)
                if self.config.c_dim >12:
                    random_pose = self.D3MM_model(real_x_random)
                    target_pose = self.change_pose_without_shape(real_pose, random_pose)
                else:
                    target_pose = Variable(self.D3MM_model(real_x_random)[:,0:self.config.c_dim].data)

                target_c = torch.cat([target_pose,fake_illumination],dim=1)
                real_c = torch.cat([real_pose,illumination_list],dim=1)
                # ================== Train D ================== #
                if (i+1) % self.config.d_train_repeat == 0:

                    # real
                    out_src , out_illumination_cls = self.D(real_x)
                    d_loss_real = - torch.mean(out_src)
                    d_loss_cls = binary_cross_loss(out_illumination_cls[self.config.batch_size:], real_label[self.config.batch_size:])

                    # fake
                    fake_x,_ = self.G_operation(real_x, target_c)
                    fake_x = Variable(fake_x.data)
                    out_src,_ = self.D(fake_x)
                    d_loss_fake = torch.mean(out_src)


                    # Backward + Optimize
                    d_loss = d_loss_real + d_loss_fake + self.config.lambda_cls * d_loss_cls
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    d_loss_gp = gradient_p(real_x, fake_x, self.D)
                    d_loss = self.config.lambda_gp * d_loss_gp
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()


                    self.loss['D_loss_real'] = -d_loss_real.item()
                    self.loss['D_loss_fake'] = d_loss_fake.item()
                    self.loss['D_loss_cls'] = self.config.lambda_cls * d_loss_cls.item()
                    self.loss['D_loss_gp'] = self.config.lambda_gp *d_loss_gp.item()

                # ================== Train G ================== #
                if (i+1) % self.config.g_train_repeat == 0:


                    fake_x,_ = self.G_operation(real_x, target_c)

                    if self.config.loss_rec and i%2==1:

                        rec_x,_ = self.G_operation(fake_x, real_c)

                        g_loss_rec = self.rec_loss(real_x , rec_x)
                        self.loss['G_loss_rec'] = self.config.lambda_rec * g_loss_rec.item()
                    else:
                        g_loss_rec = 0

                    if self.config.loss_pose_symmetry and i%2==1:
                        pose_GT_target = self.to_var(self.h_flip(real_x[:real_x.size(0)//2],3).data)
                        mask_x_sys = self.to_var(self.h_flip(mask_x[:real_x.size(0)//2],3).data)
                        hflip_pose = Variable((self.D3MM_model(pose_GT_target)[:real_x.size(0)//2, 0:self.config.c_dim]).data)

                        pose_sysx, _ = self.G_operation(real_x[:real_x.size(0)//2],torch.cat([hflip_pose,fake_illumination[:real_x.size(0)//2]],dim=1))

                        if self.config.loss_rec_with_mask:
                            plot_tensor(pose_GT_target*mask_x_sys)
                            g_loss_pose_sym = self.rec_loss(pose_sysx*mask_x_sys, pose_GT_target*mask_x_sys)
                        else:
                            g_loss_pose_sym = self.rec_loss(pose_sysx, pose_GT_target)
                        self.loss['G_loss_pose_sym'] = self.config.lambda_pose_symmetry * g_loss_pose_sym.item()
                    else:
                        g_loss_pose_sym = 0

                    target_pose__ = self.D3MM_model(fake_x)[:,0:self.config.c_dim]

                    out_src ,out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)   #real/fake
                    if i %2==0:
                        g_loss_cls = binary_cross_loss(out_cls, fake_label)
                        self.loss['G_loss_cls'] = self.config.lambda_cls * g_loss_cls.item()
                    else:
                        g_loss_cls = 0
                    if self.config.c_dim > 12:
                        g_loss_pose_l2_pose = self.l2_criterion(target_pose[:,0:12],target_pose__[:,0:12])
                        g_loss_pose_l2_shape = self.l2_criterion(target_pose[:,12:211], target_pose__[:,12:211])
                        g_loss_pose_l2_exp = self.l2_criterion(target_pose[:,221:], target_pose__[:,221:])
                        g_loss_pose_l2 = 1.0*g_loss_pose_l2_pose+0.01*g_loss_pose_l2_shape+self.config.expression_w*g_loss_pose_l2_exp
                    else:
                        g_loss_pose_l2 = self.l2_criterion(target_pose, target_pose__)
                    if self.config.loss_identity:
                        g_loss_id_l2 = self.l2_criterion(self.R(fake_x,'test',self.config.gray), self.R(real_x,'test',self.config.gray))
                        self.loss['G_g_loss_id_l2'] = self.config.lambda_id_l2 * g_loss_id_l2.item()
                    else:
                        g_loss_id_l2 = 0

                    g_loss = g_loss_fake +\
                                self.config.lambda_rec * g_loss_rec + \
                                self.config.lambda_cls * g_loss_cls + \
                                self.config.lambda_pose_l2 *  g_loss_pose_l2+ \
                                self.config.lambda_id_l2* g_loss_id_l2+\
                                self.config.lambda_pose_symmetry*g_loss_pose_sym


                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    self.loss['G_loss_fake'] = -g_loss_fake.item()
                    self.loss['G_g_loss_pose_l2'] = self.config.lambda_pose_l2*g_loss_pose_l2.item()



                if (i) % (iters_per_epoch//1000) == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.config.num_epochs, i+1, iters_per_epoch)
                    for tag, value in self.loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                        if tag in self.logger_dict.keys():

                            self.logger_dict[tag].scalar_summary('loss', value, self.display_counter)
                        else:

                            self.logger_dict[tag] = Logger(self.config.logs_path+'/'+tag)
                            self.logger_dict[tag].scalar_summary('loss', value, self.display_counter)
                    self.display_counter+=1
                    print(log)


            # Decay learning rate
            if (e+1) % 1 == 0:
                self.save_model(e)
            if (e+1) > (self.config.num_epochs - self.config.num_epochs_decay):
                g_lr -= (self.config.g_lr / float(self.config.num_epochs_decay))
                d_lr -= (self.config.d_lr / float(self.config.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
    def val(self,e):
        self.G.eval()

        for i, (real_x, img_names, ref_img) in enumerate(self.test_loader):

            real_x = self.to_var(real_x, volatile=True)
            ref_img_ = self.to_var(ref_img[0].float(), volatile=True)
            illumination_c_list = self.make_celeb_labels()
            shuffle(illumination_c_list)
            # if self.config.c_dim > 12:
            target_pose = Variable(self.D3MM_model(ref_img_)[:, 0:self.config.c_dim].data)
            # Start translations
            fake_image_list = [real_x]
            for fixed_c_index in range(len(target_pose)):

                attribute_pose = target_pose[fixed_c_index].unsqueeze(0)
                attribute_illumination = illumination_c_list[fixed_c_index].unsqueeze(0)
                attribute = torch.cat([attribute_pose, attribute_illumination], dim=1)
                attribute = attribute.expand(real_x.size(0), attribute.size(1))
                fake_image_list.append(self.G_operation(real_x, attribute)[0])
            fake_images = torch.cat(fake_image_list, dim=3)

            save_path = os.path.join(self.config.result_path, '{}_{}_fake.png'.format(i + 1, e))

            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

        self.G.train()

    def test(self, e):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        data_loader = self.test_loader
        needed_epoch = self.config.test_model.split("-")
        if len(needed_epoch) > 1:
            start_e = int(needed_epoch[0])
            end_e = int(needed_epoch[1]) + 1
        else:
            start_e = int(needed_epoch[0])
            end_e = start_e + 1
        used_illumination = [1,2,9,13,1,2,9,13,1]
        ##used_illumination = [1,8,8,8,8,8,8,8,8,8,2]
        used_poses = [1, 2, 3, 4, 5, 6, 7, 8,9]
        #used_illumination = [8]
        used_illumination = [1,2,9,13]
        for epch in range(start_e, end_e, 3):

            G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(str(epch)))
            self.G.load_state_dict(torch.load(G_path))
            self.G.eval()
            illumination_c_list = self.make_celeb_labels()

            for i, (real_x, img_names,ref_img) in enumerate(self.test_loader):

                real_x = self.to_var(real_x, volatile=True)
                ref_img_ = self.to_var(ref_img[0].float(), volatile=True)
                #if self.config.c_dim > 12:

                target_pose = Variable(self.D3MM_model(ref_img_[:])[:, 0:self.config.c_dim].data)
                real_pose = Variable((self.D3MM_model(real_x)[:, 0:self.config.c_dim]).data)


                #target_expression = Variable(self.D3MM_model(ref_img_[-2:])[:, 0:self.config.c_dim].data)
                # Start translations
                #for target_expression_idx in range(len(target_expression)):
                fake_image_lists = []
                for used_illumination_index in used_illumination:
                #used_illumination_index = 0
                #if True:
                    fake_image_list = [real_x]
                    for fixed_c_index in range(len(target_pose)):
                        attribute_pose = target_pose[fixed_c_index].unsqueeze(0)
                        #attribute_expression= target_expression[target_expression_idx].unsqueeze(0)
                        #attribute_illumination = illumination_c_list[used_illumination_index].unsqueeze(0)
                        attribute_illumination = illumination_c_list[used_illumination_index].unsqueeze(0)
                        attribute = torch.cat([attribute_pose, attribute_illumination], dim=1)
                        attribute = attribute.expand(real_x.size(0), attribute.size(1))
                        attribute = Variable(torch.cat((attribute[:, 0:12], real_pose[:, 12:211], attribute[:, 211:240],attribute[:, 240:]), 1).data)
                        fake_image_list.append(self.G_operation(real_x, attribute)[0])
                    fake_images = torch.cat(fake_image_list, dim=3)
                    fake_image_lists.append(fake_images)
                    save_path = os.path.join(self.config.result_path, '{}_{}_{}_fake.png'.format(i + 1,used_illumination_index, epch))
                    save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)

                    print('Translated test images and saved into "{}"..!'.format(save_path))

        self.G.train()

    def test_save_single_img(self):

        data_loader = self.test_loader
        needed_epoch = self.config.test_model.split("-")
        if len(needed_epoch)>1:
            start_e = int(needed_epoch[0])
            end_e = int(needed_epoch[1])+1
        else:
            start_e = int(needed_epoch[0])
            end_e = start_e+1
        for epch in range(start_e,end_e):

            # Load trained parameters
            G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(str(epch)))
            self.G.load_state_dict(torch.load(G_path))
            D_path = os.path.join(self.config.model_save_path, '{}_D.pth'.format(str(epch)))
            self.D.load_state_dict(torch.load(D_path))
            self.D.eval()
            self.G.eval()
            illumination_c_list = self.make_celeb_labels()

            k = 0
            write =[]
            for _, (imgs, img_names,ref_img) in enumerate(data_loader):

                real_x = self.to_var(imgs, volatile=True)
                ref_img = self.to_var(ref_img[0], volatile=True)

                if self.config.c_dim>12:
                    real_pose = Variable((self.D3MM_model(real_x)[:, 0:self.config.c_dim]).data)
                    real_shape = real_pose[:, 12:211]
                    target_pose = self.D3MM_model(ref_img)
                else:
                    target_pose = Variable((self.D3MM_model(ref_img)[:, 0:self.config.c_dim]).data)
                # for original data
                if not self.config.test_mode=='norm':
                    for j in range(len(real_x)):
                        img_name = img_names[j]
                        img_foler = img_name.split('/')[:-1]
                        img_foler = '/'.join(img_foler)
                        img_name = img_name.split('/')[-1]
                        save_path = self.config.test_single_path + '/' + str(epch) + '/' + img_foler
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_path = os.path.join(save_path, img_name)
                        save_image(self.denorm(real_x[j].data), save_path, nrow=1, padding=0)
                        if k % 1000 == 0:
                            print('Translated test images and saved into "{}"..number"{}" !'.format(save_path, str(k + 1)))
                        k += 1
                # sys data
                li = [i for i in range(len(target_pose))]
                used_pose = random.sample(li, self.config.use_pose_numbers)
                for fixed_c_index in used_pose:
                    if not self.config.test_mode == 'norm':
                   #used_illumination = [1 + random.randint(0, 12)]
                        li = [1, 2, 9, 13]
                        used_illumination = random.sample(li, 1)
                    else:
                        used_illumination = [8]
                    for used_illumination_index in used_illumination:

                        attribute_pose = target_pose[fixed_c_index].unsqueeze(0)
                        attribute_illumination = illumination_c_list[used_illumination_index].unsqueeze(0)

                        attribute = torch.cat([attribute_pose, attribute_illumination], dim=1)
                        attribute = attribute.expand(real_x.size(0), attribute.size(1))
                        #attribute = Variable(torch.cat(((attribute[:, 0:12]+real_pose[:, 0:12])/2, real_shape, (attribute[:, 211:240]+real_pose[:, 211:240])/2,attribute[:, 240:]), 1).data)
                        attribute = Variable(torch.cat((attribute[:, 0:12] ,
                                                        real_shape,
                                                        attribute[:, 211:240] ,
                                                        attribute[:, 240:]), 1).data)

                        fake_x = self.G_operation(real_x, attribute)[0]
                        for j in range(len(real_x)):
                            img_name = img_names[j]
                            img_foler = img_name.split('/')[:-1]
                            img_foler = '/'.join(img_foler)
                            img_name = img_name.split('/')[-1]
                            save_path = self.config.test_single_path+'/'+str(epch)+'/' + img_foler
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            if not self.config.test_mode == 'norm':
                                img_name = img_name.split('.')[0]+'_'+str(fixed_c_index)+'_'+str(used_illumination_index)+'.'+img_name.split('.')[1]

                            save_path = os.path.join(save_path, img_name)

                            #info = img_names[j]+" "+str(self.D(fake_x[j].unsqueeze(0))[0].data.cpu().numpy())
                            #write.append(info)

                            save_image(self.denorm(fake_x[j].data), save_path, nrow=1, padding=0)

                            if k%1000==0:
                                print('Translated test images and saved into "{}"..number"{}" !'.format(save_path,str(k+1)))
                            k+=1
            #from data.utils import write_txt
            #write_txt(self.config.test_single_path+'/'+str(epch)+'/'+'score.txt',write)




