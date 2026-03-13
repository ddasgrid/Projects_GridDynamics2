import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
import warnings
from contextlib import nullcontext
from pathlib import Path

warnings.filterwarnings("ignore")

# ==========================================
# 1. THE ADVANCED ARCHITECTURE BLUEPRINT
# ==========================================

# model 1

class VisualEntailmentModel1(nn.Module):
    def __init__(
        self, 
        vision_model_name='google/vit-base-patch16-224', 
        text_model_name='bert-base-uncased',
        hidden_dim=512,              # Experiment Variable (e.g., 256, 512)
        dropout_rate=0.1,            # Experiment Variable (e.g., 0.1, 0.3)
        depth=2,                     # Experiment Variable (e.g., 1, 2, 4)
        fusion_type='concat',        # 'concat', 'multiply', 'add', 'attention'
        freeze_mode='partial',       # 'full', 'none', 'partial'
        num_layers_to_freeze=9       # How many transformer layers to lock (usually out of 12)
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        # ==========================================
        # 1. LOAD PRE-TRAINED ENCODERS
        # ==========================================
        self.vit = AutoModel.from_pretrained(vision_model_name)
        self.bert = AutoModel.from_pretrained(text_model_name)
        
        # ==========================================
        # 2. APPLY FREEZING STRATEGY
        # ==========================================
        self._apply_freezing(freeze_mode, num_layers_to_freeze)

        # ==========================================
        # 3. SETUP FUSION MECHANISM SIZES
        # ==========================================
        vit_hidden = self.vit.config.hidden_size
        bert_hidden = self.bert.config.hidden_size
        
        if self.fusion_type == 'concat':
            base_fused_dim = vit_hidden + bert_hidden
            
        elif self.fusion_type in ['multiply', 'add', 'attention']:
            # These three methods require the visual and textual vectors to match in size
            if vit_hidden != bert_hidden:
                raise ValueError(f"For '{self.fusion_type}' fusion, vision and text hidden sizes must match.")
            
            base_fused_dim = vit_hidden
            
            # If attention is chosen, initialize the Multi-Head Attention layer
            if self.fusion_type == 'attention':
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=vit_hidden, 
                    num_heads=8, 
                    batch_first=True
                )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from 'concat', 'multiply', 'add', 'attention'.")

        # ==========================================
        # 4. BUILD THE FUSION HEAD (Controls Depth)
        # ==========================================
        fusion_layers = []
        in_features = base_fused_dim
        
        # Stacks layers based on your 'depth' parameter (1, 2, or 4)
        for _ in range(depth):
            fusion_layers.append(nn.Linear(in_features, hidden_dim))
            fusion_layers.append(nn.LayerNorm(hidden_dim)) 
            fusion_layers.append(nn.GELU()) 
            fusion_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim # After the first layer, the input size becomes hidden_dim
            
        self.fusion_head = nn.Sequential(*fusion_layers)

        # ==========================================
        # 5. BUILD THE CLASSIFIER HEAD
        # ==========================================
        # A strictly separated final layer that maps the mixed features to your 3 classes
        self.classifier_head = nn.Linear(hidden_dim, 3)

    def _apply_freezing(self, mode, num_layers):
        """Internal helper to lock parameters based on the chosen strategy."""
        if mode == 'full':
            print("Freezing Strategy: FULL. Locking all encoder parameters.")
            for param in self.vit.parameters(): param.requires_grad = False
            for param in self.bert.parameters(): param.requires_grad = False
                
        elif mode == 'none':
            print("Freezing Strategy: NONE. Training all parameters. (High VRAM usage)")
            pass # PyTorch defaults to requires_grad=True, so doing nothing leaves them unfrozen.
            
        elif mode == 'partial':
            print(f"Freezing Strategy: PARTIAL. Locking embeddings and first {num_layers} layers.")
            
            def freeze_n_layers(model, n):
                # 1. Freeze the word/patch embeddings
                if hasattr(model, 'embeddings'):
                    for param in model.embeddings.parameters():
                        param.requires_grad = False
                
                # 2. Freeze the specified number of transformer blocks
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                    encoder_layers = model.encoder.layer
                    # Prevent crashing if requested layers exceed actual model layers
                    n = min(n, len(encoder_layers)) 
                    for i in range(n):
                        for param in encoder_layers[i].parameters():
                            param.requires_grad = False

            freeze_n_layers(self.vit, num_layers)
            freeze_n_layers(self.bert, num_layers)

    def forward(self, pixel_values, input_ids, attention_mask):
        # ------------------------------------------
        # A. Feature Extraction
        # ------------------------------------------
        vit_outputs = self.vit(pixel_values=pixel_values)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the global [CLS] tokens
        vit_cls = vit_outputs.last_hidden_state[:, 0, :]
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]
        
        # ------------------------------------------
        # B. Base Fusion Operation
        # ------------------------------------------
        if self.fusion_type == 'concat':  #This is the clip style fusion that you started with, and it works well as a strong baseline.
            base_fused = torch.cat((vit_cls, bert_cls), dim=1)
            
        elif self.fusion_type == 'multiply':
            base_fused = torch.mul(vit_cls, bert_cls)
            
        elif self.fusion_type == 'add':
            base_fused = torch.add(vit_cls, bert_cls)
            
        elif self.fusion_type == 'attention':
            # Text queries the image patches
            query = bert_cls.unsqueeze(1) 
            key_value = vit_outputs.last_hidden_state 
            attn_output, _ = self.cross_attention(query=query, key=key_value, value=key_value)
            base_fused = attn_output.squeeze(1) 
            
        # ------------------------------------------
        # C. Deep Fusion Mixing (Depth 1, 2, or 4)
        # ------------------------------------------
        deep_fused_features = self.fusion_head(base_fused)
        
        # ------------------------------------------
        # D. Final Classification (The Verdict)
        # ------------------------------------------
        logits = self.classifier_head(deep_fused_features)
        
        return logits



# model 2


#----------------------------------------------
class SwiGLU_MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.w12 = nn.Linear(in_features, hidden_features * 2)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.dropout(hidden)
        return self.w3(hidden)

class VisualEntailmentModel(nn.Module):
    def __init__(
        self, 
        vision_model_name='google/vit-base-patch16-224', 
        text_model_name='bert-base-uncased',
        hidden_dim=512,              
        dropout_rate=0.3,            
        depth=1,                     
        fusion_type='attention',        
        freeze_mode='partial',       
        num_layers_to_freeze=10       
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        self.vit = AutoModel.from_pretrained(vision_model_name)
        self.bert = AutoModel.from_pretrained(text_model_name)
        
        self._apply_freezing(freeze_mode, num_layers_to_freeze)

        vit_hidden = self.vit.config.hidden_size
        bert_hidden = self.bert.config.hidden_size
        base_fused_dim = vit_hidden
        
        if self.fusion_type == 'attention':
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=vit_hidden, 
                num_heads=8, 
                batch_first=True
            )

        fusion_layers = []
        in_features = base_fused_dim
        for _ in range(depth):
            fusion_layers.append(nn.Linear(in_features, hidden_dim))
            fusion_layers.append(nn.LayerNorm(hidden_dim)) 
            fusion_layers.append(nn.GELU()) 
            fusion_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim 
            
        self.fusion_head = nn.Sequential(*fusion_layers)

        self.classifier_head = SwiGLU_MLP(
            in_features=hidden_dim, 
            hidden_features=hidden_dim // 2, 
            out_features=3, 
            dropout_rate=dropout_rate
        )

    def _apply_freezing(self, mode, num_layers):
        if mode == 'partial':
            def freeze_n_layers(model, n):
                if hasattr(model, 'embeddings'):
                    for param in model.embeddings.parameters(): param.requires_grad = False
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                    encoder_layers = model.encoder.layer
                    n = min(n, len(encoder_layers)) 
                    for i in range(n):
                        for param in encoder_layers[i].parameters(): param.requires_grad = False
            freeze_n_layers(self.vit, num_layers)
            freeze_n_layers(self.bert, num_layers)

    def forward(self, pixel_values, input_ids, attention_mask):
        vit_outputs = self.vit(pixel_values=pixel_values)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        vit_cls = vit_outputs.last_hidden_state[:, 0, :]
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]
        
        if self.fusion_type == 'attention':
            query = bert_cls.unsqueeze(1) 
            key_value = vit_outputs.last_hidden_state 
            attn_output, _ = self.cross_attention(query=query, key=key_value, value=key_value)
            base_fused = attn_output.squeeze(1) 
            
        deep_fused_features = self.fusion_head(base_fused)
        logits = self.classifier_head(deep_fused_features)
        return logits 

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Visual Entailment", page_icon="👁️", layout="wide")

def apply_minimal_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(56, 189, 248, 0.12), transparent 30%),
                radial-gradient(circle at 88% 20%, rgba(20, 184, 166, 0.1), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            font-family: "Avenir Next", "Segoe UI", "Trebuchet MS", sans-serif;
        }
        .main .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .ve-title {
            margin-bottom: 0.2rem;
            font-size: clamp(1.6rem, 3vw, 2.2rem);
            color: #111827;
            font-weight: 800;
            letter-spacing: -0.02em;
            animation: ve-rise 420ms ease-out both;
        }
        .ve-title-emphasis {
            background: linear-gradient(90deg, #0f766e, #2563eb);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .ve-subtitle {
            margin-bottom: 1.5rem;
            color: #334155;
            font-size: 0.96rem;
            max-width: 60ch;
            animation: ve-rise 620ms ease-out both;
        }
        div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid #dbe2ea;
            border-radius: 14px;
            padding: 0.45rem 0.55rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(4px);
            transition: transform 160ms ease, box-shadow 160ms ease;
            animation: ve-rise 460ms ease-out both;
        }
        div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.1);
        }
        .ve-card-title {
            margin: 0 0 0.15rem 0;
            color: #111827;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .ve-card-subtitle {
            margin: 0 0 0.9rem 0;
            color: #475569;
            font-size: 0.9rem;
        }
        .ve-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            border: 1px solid transparent;
        }
        .ve-badge-entailment {
            color: #166534;
            background: #dcfce7;
            border-color: #86efac;
        }
        .ve-badge-neutral {
            color: #854d0e;
            background: #fef9c3;
            border-color: #fde047;
        }
        .ve-badge-contradiction {
            color: #991b1b;
            background: #fee2e2;
            border-color: #fca5a5;
        }
        .ve-image-wrap {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 0.6rem 0 0.9rem 0;
        }
        [data-testid="stFileUploader"] section {
            border-radius: 10px;
            border: 1px dashed #c6d3e1;
            background: #f8fbff;
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid #c7d2fe;
            transition: transform 120ms ease, box-shadow 120ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 16px rgba(37, 99, 235, 0.18);
        }
        .stTextInput > div > div > input {
            background: #fbfdff;
        }
        @keyframes ve-rise {
            from {
                opacity: 0;
                transform: translateY(6px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @media (max-width: 900px) {
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
                min-height: auto;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .ve-title,
            .ve-subtitle,
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
                animation: none;
            }
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"],
            .stButton > button {
                transition: none;
            }
            div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"]:hover,
            .stButton > button:hover {
                transform: none;
                box-shadow: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_minimal_styles()
st.markdown(
    "<h1 class='ve-title'>Visual <span class='ve-title-emphasis'>Entailment</span> Analyzer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='ve-subtitle'>Upload an image and test whether a hypothesis is Entailment, Neutral, or Contradiction.</p>",
    unsafe_allow_html=True,
)

# --- A. DEFINE YOUR BUILDER FUNCTIONS ---
# Each function builds its own unique architecture and processors

def build_sota_pipeline(x):
    """Builds your winning ViT + BERT Attention model"""
    vision_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if x == 1:
    # Notice we pass the specific hyperparameters for this file
        model = VisualEntailmentModel1(
            depth=2, 
            hidden_dim=512, 
            dropout_rate=0.1, 
            fusion_type='concat', 
            freeze_mode='full', 
            num_layers_to_freeze=12
        )
    elif x == 2:
        model = VisualEntailmentModel1(
            depth=2, 
            hidden_dim=512, 
            dropout_rate=0.1, 
            fusion_type='concat', 
            freeze_mode='partial', 
            num_layers_to_freeze=10
        )
    elif x == 3:
        model = VisualEntailmentModel1(
            depth=2, 
            hidden_dim=512, 
            dropout_rate=0.1, 
            fusion_type='concat', 
            freeze_mode='partial', 
            num_layers_to_freeze=6
        )
    return model, vision_processor, tokenizer

def build_baseline_pipeline(x):
    """Builds a totally different pipeline (EXAMPLE)"""
    # 🚨 Replace this inside with your actual baseline architecture!
    vision_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Example: Maybe this one used multiply fusion and depth 3
    if x==1:
        model = VisualEntailmentModel(
            depth=1, 
            hidden_dim=512, 
            dropout_rate=0.3, 
            fusion_type='attention', 
            freeze_mode='full', 
            num_layers_to_freeze=12 
        )
    elif x==2:
        model = VisualEntailmentModel(
            depth=1, 
            hidden_dim=512, 
            dropout_rate=0.3, 
            fusion_type='attention', 
            freeze_mode='partial', 
            num_layers_to_freeze=10 
        )
    elif x==3:
        model = VisualEntailmentModel(
            depth=1, 
            hidden_dim=512, 
            dropout_rate=0.3, 
            fusion_type='attention', 
            freeze_mode='partial', 
            num_layers_to_freeze=8 
        )
    elif x==4:
        model = VisualEntailmentModel(
            depth=1, 
            hidden_dim=512, 
            dropout_rate=0.3, 
            fusion_type='attention', 
            freeze_mode='partial', 
            num_layers_to_freeze=10
        )
    elif x==5:
        model = VisualEntailmentModel(
            depth=1, 
            hidden_dim=512, 
            dropout_rate=0.3, 
            fusion_type='attention', 
            freeze_mode='full', 
            num_layers_to_freeze=12
        )
    return model, vision_processor, tokenizer

PIPELINE_REGISTRY = {
    "final_sota_visual_entailment.pth": lambda: build_baseline_pipeline(3),
    "final_sota_visual_entailment2.pth": lambda: build_baseline_pipeline(4),
    "final_sota_visual_entailment3.pth": lambda: build_baseline_pipeline(5),
    "sota_visual_entailment.pth": lambda: build_baseline_pipeline(2),
    "sota_visual_entailment2.pth": lambda: build_baseline_pipeline(1),
    "saved_model_acc_58.0.pth": lambda: build_sota_pipeline(1),
    "best_model_acc_70.7.pth": lambda: build_sota_pipeline(2),
    "best_model_acc_73.7.pth":lambda: build_sota_pipeline(3),

}

# ==========================================
# 3.2. LOAD THE BRAIN (CACHED FOR SPEED)
# ==========================================
@st.cache_resource
def load_ai_pipeline(weights_path):
    device_str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    try:
        # Look up the correct builder function
        if weights_path not in PIPELINE_REGISTRY:
            raise ValueError(f"No pipeline registered for '{weights_path}'")
            
        builder_fn = PIPELINE_REGISTRY[weights_path]
        
        # Build the exact right model and processors
        model, vision_processor, tokenizer = builder_fn()

        # Pour the weights in
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()

        return model, vision_processor, tokenizer, device, device_str, None
    except Exception as e:
        return None, None, None, None, None, str(e)


def get_weight_files():
    # Only show files in the dropdown that actually exist in your folder AND your registry
    available_files = [f for f in PIPELINE_REGISTRY.keys() if Path(f).exists()]
    return available_files

# ==========================================
# 4. FRONTEND INTERFACE
# ==========================================

labels_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
badge_class_map = {
    "Entailment": "ve-badge-entailment",
    "Neutral": "ve-badge-neutral",
    "Contradiction": "ve-badge-contradiction",
}

prediction_payload = None
notice = None
selected_weight = None
weight_files = get_weight_files()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<h2 class='ve-card-title'>1. Input</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='ve-card-subtitle'>Choose model weights, then add one image and one hypothesis.</p>",
            unsafe_allow_html=True,
        )

        if weight_files:
            selected_weight = st.selectbox(
                "Model weights",
                options=weight_files,
                index=0,
                help="Select which trained checkpoint should be used for this prediction.",
            )
            st.caption(f"Selected checkpoint: `{selected_weight}`")
        else:
            st.error("No `.pth` weight files found in this folder.")

        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown("<div class='ve-image-wrap'>", unsafe_allow_html=True)
            st.image(image, caption="Preview", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        hypothesis = st.text_input(
            "Hypothesis",
            placeholder="e.g., A dog is running in the grass.",
        )
        analyze_btn = st.button(
            "Analyze",
            use_container_width=True,
            type="primary",
            disabled=(len(weight_files) == 0),
        )

if analyze_btn:
    if not selected_weight:
        notice = ("error", "Please select a model weights file.")
    elif image is None or not hypothesis.strip():
        notice = ("warning", "Please provide both an image and a hypothesis.")
    else:
        with st.spinner("Loading model..."):
            model, processor, tokenizer, device, device_str, load_error = load_ai_pipeline(selected_weight)

        if model is None:
            notice = (
                "error",
                f"Could not load selected weights `{selected_weight}`. Error: {load_error}",
            )
        else:
            with st.spinner("Analyzing..."):
                pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
                tokens = tokenizer(
                    hypothesis,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )

                autocast_context = (
                    torch.autocast(device_type=device_str)
                    if device_str in {"cpu", "cuda"}
                    else nullcontext()
                )
                with torch.no_grad():
                    with autocast_context:
                        logits = model(
                            pixel_values,
                            tokens["input_ids"].to(device),
                            tokens["attention_mask"].to(device),
                        )
                        probs = F.softmax(logits, dim=1)[0]

                pred_idx = torch.argmax(probs).item()
                prediction = labels_map[pred_idx]
                confidence = probs[pred_idx].item() * 100
                prediction_payload = {
                    "label": prediction,
                    "confidence": confidence,
                    "weight": selected_weight,
                    "probs": {
                        "Entailment": probs[0].item(),
                        "Neutral": probs[1].item(),
                        "Contradiction": probs[2].item(),
                    },
                }

with col2:
    with st.container(border=True):
        st.markdown("<h2 class='ve-card-title'>2. Verdict</h2>", unsafe_allow_html=True)
        st.markdown("<p class='ve-card-subtitle'>Prediction and confidence breakdown.</p>", unsafe_allow_html=True)

        if notice:
            if notice[0] == "warning":
                st.warning(notice[1])
            else:
                st.error(notice[1])
        elif prediction_payload:
            label = prediction_payload["label"]
            st.markdown(
                f"<span class='ve-badge {badge_class_map[label]}'>{label}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Model used: `{prediction_payload['weight']}`")
            st.metric("Confidence", f"{prediction_payload['confidence']:.2f}%")
            st.markdown("**Probability Breakdown**")
            for prob_label, prob_value in prediction_payload["probs"].items():
                st.progress(prob_value, text=f"{prob_label}: {prob_value * 100:.1f}%")
        else:
            st.info("Run analysis to see the result.")
