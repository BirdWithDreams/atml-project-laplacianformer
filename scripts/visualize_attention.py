import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import sys
import os
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

# Adjust imports based on your repository structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pvt import PyramidVisionBackbone
from src.models.text_seq2seq import TextSeq2SeqBackbone

# ---------------------------------------------------------------------------
# VISION HOOKS
# ---------------------------------------------------------------------------
def get_vision_hook(name, attn_module, attn_type, lambda_scale=4.0, storage_dict=None):
    def hook(module, input, output):
        B, N, _ = output.shape
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim
        
        qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1] # Shapes: (B, H, N, d)
        
        if attn_type == "vanilla":
            scale = 1.0 / (head_dim ** 0.5)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            
        elif attn_type == "laplacian":
            diff = q.unsqueeze(-2) - k.unsqueeze(-3) 
            l1_dist = torch.norm(diff, p=1, dim=-1)
            attn = torch.exp(-l1_dist / lambda_scale)
            
        storage_dict[name] = attn[0].mean(dim=0).detach().cpu().numpy()
    return hook

# ---------------------------------------------------------------------------
# NLP HOOKS
# ---------------------------------------------------------------------------
def get_nlp_vanilla_hook(name, storage_dict=None):
    def hook(module, inputs, output):
        # MultiHeadAttention inputs are (q, k, v, mask)
        q_in, k_in = inputs[0], inputs[1]
        batch_size = q_in.size(0)
        
        # Replicate Q and K linear projections
        q = module.w_q(q_in).view(batch_size, -1, module.num_heads, module.d_k).transpose(1, 2)
        k = module.w_k(k_in).view(batch_size, -1, module.num_heads, module.d_k).transpose(1, 2)
        
        scale = 1.0 / (module.d_k ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        storage_dict[name] = attn[0].mean(dim=0).detach().cpu().numpy()
    return hook

def get_nlp_laplacian_hook(name, storage_dict=None):
    def hook(module, inputs, output):
        # Laplacian 1D inputs are (x, attention_mask)
        x = inputs[0]
        batch_size = x.size(0)
        
        qkv = module.qkv(x).reshape(batch_size, -1, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        
        diff = q.unsqueeze(-2) - k.unsqueeze(-3)
        l1_dist = torch.norm(diff, p=1, dim=-1)
        attn = torch.exp(-l1_dist / module.lambda_scale)
        storage_dict[name] = attn[0].mean(dim=0).detach().cpu().numpy()
    return hook

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def generate_attention_figure(args):
    attention_maps = {}
    
    if args.mode == "vision":
        print("Initializing Vision models...")
        model_vanilla = PyramidVisionBackbone(attn_type="vanilla")
        model_laplacian = PyramidVisionBackbone(attn_type="laplacian", lambda_scale=4.0)
        
        # Register Hooks
        v_attn = model_vanilla.stages[0].blocks[0].attn
        v_attn.qkv.register_forward_hook(get_vision_hook("Vanilla (Gaussian)", v_attn, "vanilla", storage_dict=attention_maps))
        
        l_attn = model_laplacian.stages[0].blocks[0].attn
        l_attn.qkv.register_forward_hook(get_vision_hook("Laplacian", l_attn, "laplacian", lambda_scale=4.0, storage_dict=attention_maps))
        
        # Load Real Data
        print(f"Loading image from: {args.data_path}")
        if not os.path.exists(args.data_path):
            print(f"Error: Could not find {args.data_path}. Using random noise instead.")
            inputs = torch.randn(1, 3, 224, 224)
        else:
            img = Image.open(args.data_path).convert("RGB")
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            inputs = transform(img).unsqueeze(0)

    elif args.mode == "text":
        print("Initializing NLP models...")
        model_vanilla = TextSeq2SeqBackbone(attn_type="vanilla", src_vocab_size=32128, tgt_vocab_size=32128)
        model_laplacian = TextSeq2SeqBackbone(attn_type="laplacian", src_vocab_size=32128, tgt_vocab_size=32128)
        
        # Register Hooks on the Encoder's Self-Attention
        v_attn = model_vanilla.transformer.encoder_layers[0].self_attn
        v_attn.register_forward_hook(get_nlp_vanilla_hook("Vanilla (Gaussian)", storage_dict=attention_maps))
        
        l_attn = model_laplacian.transformer.encoder_layers[0].self_attn
        l_attn.register_forward_hook(get_nlp_laplacian_hook("Laplacian", storage_dict=attention_maps))
        
        # Load Real Data
        print(f"Tokenizing prompt: '{args.data_path}'")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        # Ensure we have a sequence length that makes a nice heatmap
        inputs = tokenizer(args.data_path, return_tensors="pt", padding="max_length", max_length=64)["input_ids"]

    # Run Forward Passes
    print("Running forward passes...")
    if args.mode == "vision":
        _ = model_vanilla(inputs)
        _ = model_laplacian(inputs)
    elif args.mode == "text":
        dummy_decoder_input = torch.tensor([[tokenizer.pad_token_id]])
        _ = model_vanilla(inputs, dummy_decoder_input)
        _ = model_laplacian(inputs, dummy_decoder_input)
    
    # Plotting
    print("Plotting attention maps...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (title, matrix) in zip(axes, attention_maps.items()):
        # Limit the matrix size for plotting so it isn't too dense
        subset_matrix = matrix[:100, :100] if args.mode == "vision" else matrix[:64, :64]
        
        im = ax.imshow(subset_matrix, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query Index")
        fig.colorbar(im, ax=ax)
        
    plt.tight_layout()
    filename = f"attention_map_{args.mode}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Attention Maps using Real Data")
    parser.add_argument("--mode", type=str, choices=["vision", "text"], required=True, help="Modality: 'vision' or 'text'")
    parser.add_argument("--data_path", type=str, required=True, help="Path to image file (vision) OR the text string to tokenize (text)")
    
    args = parser.parse_args()
    generate_attention_figure(args)