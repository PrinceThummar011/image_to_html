import torch
from torch import nn
from transformers import CLIPModel, GPT2LMHeadModel

class ImageToHTMLModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", gpt_model_name="gpt2"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name).vision_model
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        
       
        for param in self.clip.parameters():
            param.requires_grad = False
        
     
        self.linear = nn.Linear(self.clip.config.hidden_size, self.gpt.config.n_embd)
        
        self.position_embeddings = nn.Embedding(1, self.gpt.config.n_embd)

    def forward(self, pixel_values, labels=None):
 
        image_features = self.clip(pixel_values).last_hidden_state[:, 0, :]
        
        image_features = self.linear(image_features)
        
        position_ids = torch.zeros(image_features.size(0), 1, dtype=torch.long, device=image_features.device)
        position_embeddings = self.position_embeddings(position_ids)
        image_features = image_features.unsqueeze(1) + position_embeddings
        
        if labels is not None:
            # Training mode
            outputs = self.gpt(inputs_embeds=image_features, labels=labels)
            return outputs
        else:
            # Inference mode
            max_length = 512
            outputs = self.gpt.generate(
                inputs_embeds=image_features,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            return outputs

    def generate_html(self, pixel_values, tokenizer):
        self.eval()
        with torch.no_grad():
            outputs = self(pixel_values)
        
        generated_html = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_html

class HTMLLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

