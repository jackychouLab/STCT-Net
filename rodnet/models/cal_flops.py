import time
from thop import profile
import torch

def evaluate_multi_input_model_test(model, input_shapes, device='cuda'):
    model = model.to(device)
    model.eval()

    # 创建多个 dummy 输入
    dummy_inputs = [torch.randn(shape).to(device) for shape in input_shapes]

    # 计算 GFLOPs & 参数
    macs, params = profile(model, inputs=tuple(dummy_inputs), verbose=False)
    gflops = macs / 1e9
    mparams = params / 1e6

    # 计算 FPS
    with torch.no_grad():
        warmup = 10
        iters = 100
        # 预热
        for _ in range(warmup):
            _ = model(*dummy_inputs)
        # 正式计时
        start = time.time()
        for _ in range(iters):
            _ = model(*dummy_inputs)
        end = time.time()

        avg_infer_time = (end - start) / iters
        fps = 1 / avg_infer_time * 16


def evaluate_multi_input_model_print(model, input_shapes, device='cuda'):
    model = model.to(device)
    model.eval()

    # 创建多个 dummy 输入
    dummy_inputs = [torch.randn(shape).to(device) for shape in input_shapes]

    # 计算 GFLOPs & 参数
    macs, params = profile(model, inputs=tuple(dummy_inputs), verbose=False)
    gflops = macs / 1e9
    mparams = params / 1e6

    # 计算 FPS
    with torch.no_grad():
        warmup = 10
        iters = 100
        # 预热
        for _ in range(warmup):
            _ = model(*dummy_inputs)
        # 正式计时
        start = time.time()
        for _ in range(iters):
            _ = model(*dummy_inputs)
        end = time.time()

        avg_infer_time = (end - start) / iters
        fps = 1 / avg_infer_time * 16

    # 输出结果
    print(f'Input shapes: {input_shapes}')
    print(f'GFLOPs: {gflops:.6f}')
    print(f'Params: {mparams:.6f} M')
    print(f'Infer time: {avg_infer_time:.6f} s')
    print(f'FPS: {fps:.6f} frames/sec')


input_shapes_32 = [
        (1, 2, 16, 32, 128, 128)
    ]

input_shapes_1 = [
    (1, 2, 16, 128, 128)
]

# from rodnet_cdc_v2 import RODNetCDCDCN
# from rodnet_hg_v2 import RODNetHGDCN
# from rodnet_hgwi_v2 import RODNetHGwIDCN
# from rodnet_e import E_RODNet
# from rodnet_t import T_RODNet
# from rodnet_dcsn import RODNet_DCSN
from rodnet_myNet import myNet

model = myNet((32, 32), 1, 5, 'multi', (3, 3, 3), 'gn', 'gelu', True)
evaluate_multi_input_model_test(model, input_shapes_32)
evaluate_multi_input_model_print(model, input_shapes_32)
# model = RODNetCDCDCN(32, 1, (32, 32), False)
# evaluate_multi_input_model_test(model, input_shapes_32)
# evaluate_multi_input_model_print(model, input_shapes_32)
# model = RODNetHGDCN(32, 1, 1, (32, 32), False)
# evaluate_multi_input_model_test(model, input_shapes_32)
# evaluate_multi_input_model_print(model, input_shapes_32)
# model = RODNetHGwIDCN(32, 1, 1, (32, 32), False)
# evaluate_multi_input_model_test(model, input_shapes_32)
# evaluate_multi_input_model_print(model, input_shapes_32)
# model = E_RODNet((32, 32), 1)
# evaluate_multi_input_model_test(model, input_shapes_32)
# evaluate_multi_input_model_print(model, input_shapes_32)
# model = T_RODNet(1,  embed_dim=64, win_size=4)
# evaluate_multi_input_model_test(model, input_shapes_1)
# evaluate_multi_input_model_print(model, input_shapes_1)
# model = RODNet_DCSN(1)
# evaluate_multi_input_model_test(model, input_shapes_1)
# evaluate_multi_input_model_print(model, input_shapes_1)

