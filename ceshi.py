import torch
import h5py
import os
import numpy as np

# 导入你的模型和保存工具
from models import VariationalNetwork
from misc_utils import save_recon

def test_fastmri_single_slice():
    # ================= 1. 配置路径 =================
    # 你跑完的 checkpoint 路径
    ckpt_path = "/home/liujunda/pytorch_mri_variationalnetwork-master/lightning_logs/version_7/checkpoints/epoch=9-step=7680.ckpt"
    
    # 替换为你实际的第一个 .h5 数据和对应的敏感度图路径
    data_path = "/home/liujunda/data_fastmri_train/multicoil_train/file_brain_AXFLAIR_200_6002452.h5" 
    smap_path = "/home/liujunda/data_fastmri_train_smaps/file_brain_AXFLAIR_200_6002452_smaps.h5" 
    
    slice_idx = 8 # 选定第 8 层切片
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 2. 加载模型 =================
    model = VariationalNetwork.load_from_checkpoint(ckpt_path, save_dir="./sample_recon")
    model.eval()
    model.to(device)

    # ================= 3. 数据加载与预处理 (完全复刻 FastMRISpecificDataset) =================
    print(f"正在加载数据: {data_path} (切片 {slice_idx})...")
    with h5py.File(data_path, 'r') as f_orig, h5py.File(smap_path, 'r') as f_smap:
        k_volume = torch.tensor(f_orig['kspace'][:], dtype=torch.complex64)
        k_slice = k_volume[slice_idx]
        target = torch.tensor(f_orig['reconstruction_rss'][slice_idx], dtype=torch.float32) 
        smaps = torch.tensor(f_smap['smaps'][slice_idx], dtype=torch.complex64)

        # 计算特有的归一化系数 paper_norm
        n_sl = k_volume.shape[0]
        f_volume_norm = torch.linalg.norm(k_volume)
        paper_norm = (torch.sqrt(torch.tensor(n_sl * 1.0)* 10000.0) ) / (f_volume_norm + 1e-8)

    # 生成与训练完全一致的固定掩膜 (Fixed Mask)
    mask_tmp = torch.zeros((1, 640, 320), dtype=torch.complex64)
    num_low = int(320 * 0.08)
    pad = (320 - num_low + 1) // 2
    mask_tmp[:, :, pad : pad + num_low] = 1

    torch.manual_seed(42) 
    high_mask = torch.rand(1, 1, 320) < (0.25 - (num_low/320))
    mask = torch.logical_or(mask_tmp.bool(), high_mask).to(torch.complex64)

    # 欠采样操作
    f = k_slice * mask
    
    # 图像域转换 (Adjoint 运算)
    Finv = torch.fft.ifftshift(f * mask, dim=(-2, -1))
    Finv = torch.fft.ifft2(Finv, norm="ortho")
    Finv = torch.fft.fftshift(Finv, dim=(-2, -1))
    input0 = torch.sum(Finv * torch.conj(smaps), dim=-3)

    # 应用归一化
    f_norm = f * paper_norm
    input0_norm = input0 * paper_norm
    target_norm = target * paper_norm

    # 转为 VarNet 要求的 view_as_real 格式并增加 Batch 维度 (unsqueeze)
    u_t_in = torch.view_as_real(input0_norm.to(torch.complex64)).unsqueeze(0).to(device)
    f_in = torch.view_as_real(f_norm.to(torch.complex64)).unsqueeze(0).to(device)
    c_in = torch.view_as_real(smaps.to(torch.complex64)).unsqueeze(0).to(device)
    mask_in = mask.real.to(torch.float32).squeeze(0).unsqueeze(0).to(device)

    # ================= 4. 模型推理 =================
    print("正在运行模型推理...")
    
    # 将四个变量打包成模型需要的字典格式
    model_inputs = {
        'u_t': u_t_in,
        'f': f_in,
        'coil_sens': c_in,
        'sampling_mask': mask_in
    }
    
    with torch.no_grad():
        # 将打包好的字典传给模型
        output = model(model_inputs)

    # ================= 5. 生成并保存 1x4 对比图 =================
    print("正在生成保存照片...")
    from pathlib import Path
    
    # 你的 save_recon 函数要求 reference 也是 (N,H,W,2) 的复数张量格式
    reference_in = torch.view_as_real(target_norm.to(torch.complex64)).unsqueeze(0).to(device)

    # 【关键修复：中心裁剪逻辑】
    # 将模型输出(output)和欠采样图(u_t_in)从 640x320 裁剪到 target 的尺寸 320x320
    h_recon, w_recon = output.shape[1], output.shape[2]
    h_ref, w_ref = reference_in.shape[1], reference_in.shape[2]
    h_start = (h_recon - h_ref) // 2
    w_start = (w_recon - w_ref) // 2
    
    # 执行裁剪
    output_cropped = output[:, h_start:h_start+h_ref, w_start:w_start+w_ref, :]
    u_t_in_cropped = u_t_in[:, h_start:h_start+h_ref, w_start:w_start+w_ref, :]

    # 你的代码要求 save_dir 必须是 pathlib.Path 对象
    out_dir = Path("./sample_recon")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 将裁剪后的张量丢给 save_recon 函数
    save_recon(
        under=u_t_in_cropped,   # 裁剪后的欠采样输入 
        recon=output_cropped,   # 裁剪后的网络重建输出 
        reference=reference_in, # 全采样参考 (本身就是 320x320)
        index=9,                # 这里传切片号 8，它内部会自动把文件命名为 8.png
        save_dir=out_dir,       # 保存路径 (Path 对象)
        error_scale=10,         # 误差图放大倍数，默认给个 10 让误差更明显
        do_save=True
    )
    
    print(f"测试完成！一行四列对比图已保存至 {out_dir}/8.png")

if __name__ == "__main__":
    test_fastmri_single_slice()
