import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from fft_utils import *
from misc_utils import *
import copy
from pathlib import Path
from optimizer import IIPG
from skimage.metrics import structural_similarity as ssim_func
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

DEFAULT_OPTS = {'kernel_size':11,
                'features_in':1,
                'features_out':24,
                'do_prox_map':True,
                'pad':11,
                'vmin':-1.0,'vmax':1.0,
                'lamb_init':1.0,
                'num_act_weights':31,
                'init_type':'linear',
                'init_scale':0.04,
                'sampling_pattern':'cartesian',
                'num_stages':10,
                'seed':1,
                'optimizer':'adam','lr':1e-4,
                'activation':'rbf',
                'loss_type':'complex', 
                'momentum':0.,
                'error_scale':10,
                'loss_weight':1}

# ======= 请将这段代码添加到 models.py 中 VariationalNetwork 类的上方 =======

def tf_symmetric_pad(x, pad):
    """精确复现 TensorFlow 的 SYMMETRIC 填充"""
    # 处理高 (H)
    x = torch.cat([x[:, :, 0:pad].flip(2), x, x[:, :, -pad:].flip(2)], dim=2)
    # 处理宽 (W)
    x = torch.cat([x[:, :, :, 0:pad].flip(3), x, x[:, :, :, -pad:].flip(3)], dim=3)
    return x

@torch.no_grad()
def apply_proximal_maps(model):
    for cell in model.cell_list:
        cell.lamb.clamp_(min=0.0)

        k = cell.conv_kernel
        # 1. 独立计算实部和虚部的均值并减去 (剔除 dim=4)
        k_mean = k.mean(dim=(1, 2, 3), keepdim=True)
        k.sub_(k_mean)
        
        # 2. 结合实部和虚部计算复数总范数 (保留 dim=(1,2,3,4))
        norm = torch.sqrt(torch.sum(k**2, dim=(1, 2, 3, 4), keepdim=True))
        scale = torch.clamp(norm, min=1.0)
        k.div_(scale)

# =========================================================================

class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """ Forward pass for RBF activation

        Parameters:
        ----------
        ctx: 
        input: torch tensor (NxCxHxW)
            input tensor
        w: torch tensor (1 x C x 1 x 1 x # of RBF kernels)
            weight of the RBF kernels
        mu: torch tensor (# of RBF kernels)
            center of the RBF
        sigma: torch tensor (1)
            std of the RBF

        Returns:
        ----------
        torch tensor: linear weight combination of RBF of input
        """
        num_act_weights = w.shape[-1]
        output = input.new_zeros(input.shape)
        rbf_grad_input = input.new_zeros(input.shape)
        for i in range(num_act_weights):
            tmp = w[:,:,:,:,i] * torch.exp(-torch.square(input - mu[i])/(2* sigma ** 2))
            output += tmp
            rbf_grad_input += tmp*(-(input-mu[i]))/(sigma**2)
        del tmp
        ctx.save_for_backward(input,w,mu,sigma,rbf_grad_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors
        num_act_weights = w.shape[-1]

        #if ctx.needs_input_grad[0]:
        grad_input = grad_output * rbf_grad_input

        #if ctx.need_input_grad[1]:
        grad_w = w.new_zeros(w.shape)
        for i in range(num_act_weights):
            tmp = (grad_output*torch.exp(-torch.square(input-mu[i])/(2*sigma**2))).sum((0,2,3))
            grad_w[:,:,:,:,i] = tmp.view(w.shape[0:-1])
    
        return grad_input, grad_w, None, None


class RBFActivation(nn.Module):
    """ RBF activation function with trainable weights """
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        x_0 = np.linspace(kwargs['vmin'],kwargs['vmax'],kwargs['num_act_weights'],dtype=np.float32)
        mu = np.linspace(kwargs['vmin'],kwargs['vmax'],kwargs['num_act_weights'],dtype=np.float32)
        grid_spacing = (kwargs['vmax'] - kwargs['vmin']) / (kwargs['num_act_weights'] - 1)
        self.sigma = torch.tensor(grid_spacing, dtype=torch.float32)
        if kwargs['init_type'] == 'linear':
            w_0 = kwargs['init_scale']*x_0
        elif kwargs['init_type'] == 'tv':
            w_0 = kwargs['init_scale'] * np.sign(x_0)
        elif kwargs['init_type'] == 'relu':
            w_0 = kwargs['init_scale'] * np.maximum(x_0, 0)
        elif kwargs['init_type'] == 'student-t':
            alpha = 100
            w_0 = kwargs['init_scale'] * np.sqrt(alpha)*x_0/(1+0.5*alpha*x_0**2)
        else:
            raise ValueError("init_type '%s' not defined!" % kwargs['init_type'])
        w_0 = np.reshape(w_0,(1,1,1,1,kwargs['num_act_weights']))
        w_0 = np.repeat(w_0,kwargs['features_out'],1)
        self.w = torch.nn.Parameter(torch.from_numpy(w_0))
        self.mu = torch.from_numpy(mu)
        self.rbf_act = RBFActivationFunction.apply

    def forward(self,x):
        # x = x.unsqueeze(-1)
        # x = x.repeat((1,1,1,1,self.mu.shape[-1]))
        # if not self.mu.device == x.device:
        #     self.mu = self.mu.to(x.device)
        #     self.std = self.std.to(x.device)
        # gaussian = torch.exp(-torch.square(x - self.mu)/(2*self.std ** 2))
        # weighted_gaussian = self.w_0 * gaussian
        # out = torch.sum(weighted_gaussian,axis=-1,keepdim=False)
        if not self.mu.device == x.device:
            self.mu = self.mu.to(x.device)
            self.sigma = self.sigma.to(x.device)

        # out = torch.zeros(x.shape,dtype=torch.float32,device=x.device)
        # for i in range(self.options['num_act_weights']):
        # 	out += self.w_0[:,:,:,:,i] * torch.exp(-torch.square(x - self.mu[:,:,:,:,i])/(2*self.std ** 2))
        output = self.rbf_act(x,self.w,self.mu,self.sigma)
        	
        return output

class VnMriReconCell(nn.Module):
    """ One cell of variational network """
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        conv_kernel = np.random.randn(options['features_out'],options['features_in'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*options['features_in'])
        conv_kernel -= np.mean(conv_kernel, axis=(1,2,3,4),keepdims=True)
        conv_kernel = torch.from_numpy(conv_kernel)
        if options['do_prox_map']:
            conv_kernel = zero_mean_norm_ball(conv_kernel,axis=(1,2,3,4))


        self.conv_kernel = torch.nn.Parameter(conv_kernel)

        if self.options['activation'] == 'rbf':
            self.activation = RBFActivation(**options)
        elif self.options['activation'] == 'relu':
            self.activation = torch.nn.ReLU()
        self.lamb = torch.nn.Parameter(torch.tensor(options['lamb_init'],dtype=torch.float32))


    def mri_forward_op(self, u, coil_sens, sampling_mask, os=False):
        """
        Forward pass with kspace
        (2X the size)
        
        Parameters:
        ----------
        u: torch tensor NxHxWx2
            complex input image
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding

        Returns:
        -----------
        kspace of u with applied coil sensitivity and sampling mask
        """
        if os:
            pad_u = torch.tensor((sampling_mask.shape[1]*0.25 + 1),dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1]*0.25 - 1),dtype=torch.int16)
            u_pad = F.pad(u,[0,0,0,0,pad_u,pad_l])
        else:
            u_pad = u
        u_pad = u_pad.unsqueeze(1)
        coil_imgs = complex_mul(u_pad, coil_sens) # NxCxHxWx2
        
        Fu = fftc2d(coil_imgs) #
        
        mask = sampling_mask.unsqueeze(1) # Nx1xHxW
        mask = mask.unsqueeze(4) # Nx1xHxWx1
        mask = mask.repeat([1,1,1,1,2]) # Nx1xHxWx2

        kspace = mask*Fu # NxCxHxWx2
        return kspace

    def mri_adjoint_op(self, f, coil_sens, sampling_mask, os=False):
        """
        Adjoint operation that convert kspace to coil-combined under-sampled image
        by using coil_sens and sampling mask
        
        Parameters:
        ----------
        f: torch tensor NxCxHxWx2
            multi channel kspace
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding
        Returns:
        -----------
        Undersampled, coil-combined image
        """
        
        # Apply mask and perform inverse centered Fourier transform
        mask = sampling_mask.unsqueeze(1) # Nx1xHxW
        mask = mask.unsqueeze(4) # Nx1xHxWx1
        mask = mask.repeat([1,1,1,1,2]) # Nx1xHxWx2

        Finv = ifftc2d(mask*f) # NxCxHxWx2
        # multiply coil images with sensitivities and sum up over channels
        img = torch.sum(complex_mul(Finv,conj(coil_sens)),1)

        if os:
            # Padding to remove FE oversampling
            pad_u = torch.tensor((sampling_mask.shape[1]*0.25 + 1),dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1]*0.25 - 1),dtype=torch.int16)
            img = img[:,pad_u:-pad_l,:,:]
            
        return img

    def forward(self, inputs):
        u_t_1 = inputs['u_t'] #NxHxWx2
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']

        u_t_1 = u_t_1.unsqueeze(1) #Nx1xHxWx2
        # pad the image to avoid problems at the border
        pad = self.options['pad']
        u_t_real = u_t_1[:,:,:,:,0]
        u_t_imag = u_t_1[:,:,:,:,1]
        
        u_t_real = tf_symmetric_pad(u_t_real, pad)
        u_t_imag = tf_symmetric_pad(u_t_imag, pad)
        # split the image in real and imaginary part and perform convolution
        u_k_real = F.conv2d(u_t_real,self.conv_kernel[:,:,:,:,0],stride=1,padding=5)
        u_k_imag = F.conv2d(u_t_imag,self.conv_kernel[:,:,:,:,1],stride=1,padding=5)
        # add up the convolution results
        u_k = u_k_real + u_k_imag
        #apply activation function
        f_u_k = self.activation(u_k)
        # perform transpose convolution for real and imaginary part
        u_k_T_real = F.conv_transpose2d(f_u_k,self.conv_kernel[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag = F.conv_transpose2d(f_u_k,self.conv_kernel[:,:,:,:,1],stride=1,padding=5)

        #Rebuild complex image
        u_k_T_real = u_k_T_real.unsqueeze(-1)
        u_k_T_imag = u_k_T_imag.unsqueeze(-1)
        u_k_T =  torch.cat((u_k_T_real,u_k_T_imag),dim=-1)

        #Remove padding and normalize by number of filter
        Ru = u_k_T[:,0,pad:-pad,pad:-pad,:] #NxHxWx2
        Ru /= self.options['features_out']

        if self.options['sampling_pattern'] == 'cartesian':
            os = False
        elif not 'sampling_pattern' in self.options or self.options['sampling_pattern'] == 'cartesian_with_os':
            os = True

        Au = self.mri_forward_op(u_t_1[:,0,:,:,:],c,m,os)
        At_Au_f = self.mri_adjoint_op(Au - f, c, m,os)
        Du = At_Au_f * self.lamb
        u_t = u_t_1[:,0,:,:,:] - Ru - Du
        output = {'u_t':u_t,'f':f,'coil_sens':c,'sampling_mask':m}
        return output #NxHxWx2

class VariationalNetwork(pl.LightningModule):   
    def __init__(self,**kwargs):
        super().__init__()
        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        cell_list = []
        for i in range(options['num_stages']):
            cell_list.append(VnMriReconCell(**options))

        self.cell_list = nn.Sequential(*cell_list)
        self.log_img_count = 0

        self.step_metrics = {'loss': [], 'mse': [], 'ssim': [], 'psnr': []}
        self.epoch_history = {'loss': [], 'mse': [], 'ssim': [], 'psnr': []}

        import os
        self.mse_history = []
        self.ssim_history = []
        # 创建专门存重建结果图的文件夹
        self.image_out_dir = os.path.join(self.options['save_dir'], 'saved_images')
        os.makedirs(self.image_out_dir, exist_ok=True)


    def forward(self,inputs):
        output = self.cell_list(inputs)
        return output['u_t']
    
    def training_step(self, batch, batch_idx):
        import os
        import matplotlib.pyplot as plt
        
        recon_img = self(batch)
        ref_img = batch['reference']
        
        # ==== 中心裁剪逻辑 ====
        h_recon, w_recon = recon_img.shape[1], recon_img.shape[2]
        h_ref, w_ref = ref_img.shape[1], ref_img.shape[2]
        h_start = (h_recon - h_ref) // 2
        w_start = (w_recon - w_ref) // 2
        recon_img = recon_img[:, h_start:h_start+h_ref, w_start:w_start+w_ref, :]
        
        if self.options['loss_type'] == 'complex':
            loss = F.mse_loss(recon_img, ref_img)
        elif self.options['loss_type'] == 'magnitude':
            recon_img_mag = torch_abs(recon_img)
            ref_img_mag = torch_abs(ref_img)    
            loss = F.mse_loss(recon_img_mag, ref_img_mag)
            
        total_loss = self.options['loss_weight'] * loss
        
        # ==== 计算 MSE 和 SSIM ====
        recon_img_mag = torch_abs(recon_img)
        ref_img_mag = torch_abs(ref_img)
        with torch.no_grad():
            from skimage.metrics import structural_similarity as ssim_func
            recon_mag_np = recon_img_mag.squeeze().cpu().detach().numpy()
            ref_mag_np = ref_img_mag.squeeze().cpu().numpy()
            data_range = ref_mag_np.max() - ref_mag_np.min()
            ssim_val = ssim_func(ref_mag_np, recon_mag_np, data_range=data_range)
            
            # 记录历史数据
            self.mse_history.append(loss.item() * 100)
            self.ssim_history.append(ssim_val * 100)

            # ---------------------------------------------------------
            # 1. 实时更新曲线 (每 10 个 batch 更新一次，避免影响训练速度)
            # ---------------------------------------------------------
            if batch_idx % 10 == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # --- MSE 曲线 ---
                ax1.plot(self.mse_history, color='red')
                ax1.set_title(f"Epoch {self.current_epoch} | MSE (x10^-2): {loss.item() * 100:.4f}")
                ax1.set_yscale('log')  # 恢复对数坐标轴，和你的参考图保持一致
                
                # --- SSIM 曲线 ---
                ax2.plot(self.ssim_history, color='blue')
                ax2.set_title(f"Epoch {self.current_epoch} | SSIM (x10^-2): {ssim_val * 100:.2f}")
                # SSIM 保持普通的线性坐标轴即可
                
                plt.tight_layout()
                curve_path = os.path.join(self.options['save_dir'], 'realtime_curves.png')
                plt.savefig(curve_path)
                plt.close(fig) # 释放内存
            # ---------------------------------------------------------
            # 2. 严格按要求拼接并保存图像 (直接存进独立文件夹)
            # ---------------------------------------------------------
            if batch_idx % (int(200/self.options['batch_size']/4)) == 0:
                # 提取欠采样图并裁剪
                u_t_cropped = batch['u_t'][:, h_start:h_start+h_ref, w_start:w_start+w_ref, :]
                under_mag_np = torch_abs(u_t_cropped).squeeze().cpu().numpy()

                # 计算误差图 (差异点图)
                error_scale = self.options['error_scale']
                diff_np = np.abs(ref_mag_np - recon_mag_np) * error_scale

                # 按顺序拼接：[欠采样图, 重建图, 原图, 差异点图]
                combined_img = np.concatenate((under_mag_np, recon_mag_np, ref_mag_np, diff_np), axis=1)

                # 归一化并保存图片
                combined_img_norm = (255 * (combined_img / (combined_img.max() + 1e-8))).astype(np.uint8)
                save_path = os.path.join(self.image_out_dir, f"epoch_{self.current_epoch}_batch_{batch_idx}.png")
                plt.imsave(save_path, combined_img_norm, cmap='gray')

        self.log('train/mse', loss, prog_bar=True)
        self.log('train/ssim', ssim_val, prog_bar=True)

        return {'loss': total_loss}



    # 修改 models.py 中的 training_step
    def training_step(self, batch, batch_idx):
        recon_img = self(batch)       # 形状: [B, H_recon, W_recon, 2]
        ref_img = batch['reference']  # 形状: [B, H_ref, W_ref, 2]
        
        # 1. 计算复数模长
        recon_mag = torch.sqrt(recon_img[..., 0]**2 + recon_img[..., 1]**2 + 1e-12) # [B, 640, 320]
        ref_mag = torch.sqrt(ref_img[..., 0]**2 + ref_img[..., 1]**2 + 1e-12)       # [B, 320, 320]
        
        # ==============================================================
        # 新增：动态中心裁剪逻辑，将 640 裁剪至与标签相同的 320，防止过采样导致的尺寸报错
        # ==============================================================
        if recon_mag.shape[1] != ref_mag.shape[1]:
            diff_h = recon_mag.shape[1] - ref_mag.shape[1]
            start_h = diff_h // 2
            end_h = start_h + ref_mag.shape[1]
            recon_mag = recon_mag[:, start_h:end_h, :]

        if recon_mag.shape[2] != ref_mag.shape[2]:
            diff_w = recon_mag.shape[2] - ref_mag.shape[2]
            start_w = diff_w // 2
            end_w = start_w + ref_mag.shape[2]
            recon_mag = recon_mag[:, :, start_w:end_w]
        # ==============================================================

        # 2. 计算 Loss (此时维度完美对齐 [B, 320, 320])
        spatial_loss = torch.sum((recon_mag - ref_mag) ** 2, dim=(1, 2))
        loss = torch.mean(spatial_loss) * self.options['loss_weight']
        
        # 3. 将 Tensor 转换为 Numpy 以计算图像指标 (下方的 NumPy 计算也同步安全了)
        r_np = recon_mag.detach().cpu().numpy()
        t_np = ref_mag.detach().cpu().numpy()
        
        batch_mse, batch_ssim, batch_psnr = 0.0, 0.0, 0.0
        B = r_np.shape[0]
        for i in range(B):
            r_img = r_np[i]
            t_img = t_np[i]
            drange = t_img.max() - t_img.min()
            if drange == 0: drange = 1.0 
            
            batch_mse += np.mean((r_img - t_img)**2)
            batch_ssim += ssim(t_img, r_img, data_range=drange)
            batch_psnr += psnr(t_img, r_img, data_range=drange)
            
        self.step_metrics['loss'].append(loss.item())
        self.step_metrics['mse'].append(batch_mse / B)
        self.step_metrics['ssim'].append(batch_ssim / B)
        self.step_metrics['psnr'].append(batch_psnr / B)

        # 后面原有的图片记录保持不变...

        # 5. TensorBoard 原有图片记录逻辑
        if batch_idx % (int(200/self.options['batch_size']/4)) == 0:
            sample_img = save_recon(batch['u_t'],recon_img,ref_img,batch_idx,'save_dir',self.options['error_scale'],False)
            sample_img = sample_img[np.newaxis,:,:]
            self.logger.experiment.add_image('sample_recon',sample_img.astype(np.uint8),self.log_img_count)
            self.log_img_count += 1

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def on_epoch_end(self):
        """每个 Epoch 结束时，汇总平均指标，并更新曲线图"""
        if len(self.step_metrics['loss']) == 0:
            return
            
        # 汇总当前 Epoch 的平均指标
        self.epoch_history['loss'].append(np.mean(self.step_metrics['loss']))
        self.epoch_history['mse'].append(np.mean(self.step_metrics['mse']))
        self.epoch_history['ssim'].append(np.mean(self.step_metrics['ssim']))
        self.epoch_history['psnr'].append(np.mean(self.step_metrics['psnr']))
        
        # 清空 step 记录，为下一个 Epoch 做准备
        self.step_metrics = {'loss': [], 'mse': [], 'ssim': [], 'psnr': []}
        
        # 生成并保存四联图
        self.plot_training_curves()

    def plot_training_curves(self):
        """生成 Loss, MSE, SSIM, PSNR 四张并排的曲线图"""
        epochs = range(1, len(self.epoch_history['loss']) + 1)
        fig, axs = plt.subplots(1, 4, figsize=(24, 5))

        # 1. Loss 曲线
        axs[0].plot(epochs, self.epoch_history['loss'], 'b-', linewidth=2)
        axs[0].set_title('Loss Curve', fontsize=16)
        axs[0].set_xlabel('Epoch', fontsize=14)
        axs[0].grid(True, linestyle='--', alpha=0.7)

        # 2. MSE 曲线
        axs[1].plot(epochs, self.epoch_history['mse'], 'r-', linewidth=2)
        axs[1].set_title('MSE', fontsize=16)
        axs[1].set_xlabel('Epoch', fontsize=14)
        axs[1].grid(True, linestyle='--', alpha=0.7)

        # 3. SSIM 曲线
        axs[2].plot(epochs, self.epoch_history['ssim'], 'g-', linewidth=2)
        axs[2].set_title('SSIM', fontsize=16)
        axs[2].set_xlabel('Epoch', fontsize=14)
        axs[2].grid(True, linestyle='--', alpha=0.7)

        # 4. PSNR 曲线
        axs[3].plot(epochs, self.epoch_history['psnr'], 'm-', linewidth=2)
        axs[3].set_title('PSNR', fontsize=16)
        axs[3].set_xlabel('Epoch', fontsize=14)
        axs[3].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        
        # 图片将保存在你的 exp/basic_varnet 目录下
        save_path = Path(self.options['save_dir']) / 'metrics_curves.png'
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        PyTorch Lightning 钩子：在每个 Batch 的梯度反向传播和优化器 step() 之后自动调用。
        此时参数刚刚被 Adam/SGD 篡改，我们需要立刻把它们拉回约束空间。
        """
        apply_proximal_maps(self)

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}
    

    def configure_optimizers(self):
        if self.options['optimizer'] == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.options['lr'])
        elif self.options['optimizer'] == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.options['lr'], momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.options['lr'], momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'iipg':
            # 实例化自定义优化器
            iipg_optimizer = IIPG(
                optimizer=torch.optim.SGD, 
                params=self.parameters(), 
                lr=self.options['lr'], 
                momentum=self.options['momentum']
            )
            # 显式打包返回，确保 Lightning 兼容性
            return {
                "optimizer": iipg_optimizer
            }
