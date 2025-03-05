import torch
import torch.nn as nn
import torch.nn.init as init
from tensordict.nn.distributions import NormalParamExtractor

class SimpleMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            n_hidden: int,
            output_dim: int,
            is_value_fn: bool = False,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        super(SimpleMLP, self).__init__()
        self.in_dim = input_dim
        self.n_hidden = n_hidden
        self.out_dim = output_dim
        self.gpu_device = device
        self.seq = nn.Sequential(
            nn.Linear(input_dim, n_hidden, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, 2 * output_dim if not is_value_fn else output_dim, device=device),
            NormalParamExtractor() if not is_value_fn else nn.Identity(),
        )

    def forward(self, x):        
        # Check if the input tensor has the right number of sequences
        if x.shape[-1] != self.in_dim:
            # Add zeros to the tensor where x has size (batch_size, seq_len, state_dim)
            x = torch.cat([x, torch.zeros(self.in_dim - x.shape[-1]).to(self.gpu_device)], dim=-1)

        x = self.seq(x)

        return x

class FloatEmbedder(nn.Module):
    """"
    Class to embed the float (continuous) values of the states and actions. Child class of nn.Module.
    """
    def __init__(
            self,
            input_dim: int,
            embed_dim: int,
            dropout: float
        ):
        super(FloatEmbedder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)
        
        layers = []
        layers.append(nn.Linear(input_dim, self.embed_dim // 4))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.embed_dim // 4, self.embed_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.embed_dim // 2, self.embed_dim))
        layers.append(nn.LayerNorm(self.embed_dim))
        self.embed = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for layer in self.embed:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        return x
    
class PositionalEncoder(nn.Module):
    """
    Class to encode the position of the states and actions using the Attention Is All You Need functions. Child class of nn.Module.
    """
    def __init__(
            self,
            max_len: int,
            d_model: int,
            dropout: float
        ):
        super(PositionalEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.empty((0, d_model), dtype=torch.float32, requires_grad=True)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0, requires_grad=True)) / d_model))

        for pos in range(max_len):
            sines = torch.sin(pos * div_term)
            cosines = torch.cos(pos * div_term)

            interleaved = torch.stack((sines, cosines), dim=1).flatten()

            pe = torch.cat((pe, interleaved.unsqueeze(0)), dim=0)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.pe = pe

    def forward(self, x):
        x = (x + self.pe[:x.size(0), :])
        x = self.dropout(x)
        return x
    
class SegmentPositionalEncoder(nn.Module):
    """
    Class to encode the position of the states and actions using the a concatenated tensor which represents the sequence position. Child class of nn.Module.
    """
    def __init__(
            self,
            max_len: int,
            d_model: int,
            encoding_segment_size: int,
            dropout: float
        ):
        super(SegmentPositionalEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.encoding_segment_size = encoding_segment_size
        self.embed = nn.Embedding(max_len, encoding_segment_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        x = torch.cat([x, self.embed(positions)], dim=-1)
        x = self.dropout(x)
        return x
    
class EOSTransformerEncoder(nn.TransformerEncoder):
    """
    Class to create a transformer encoder for the Earth Observation Satellite model. Child class of nn.TransformerEncoder.
    """
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            activation: str,
            batch_first: bool = True,
            kaiming_init: bool = False
        ):

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )

        super(EOSTransformerEncoder, self).__init__(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.architecture_type = "TransformerEncoder"

        if kaiming_init:
            self.override_linears_with_kaiming()

    def override_linears_with_kaiming(self):
        for layer in self.layers:
            for module in layer.children():
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    init.zeros_(module.bias)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

class EOSTransformer(nn.Transformer):
    """
    Class to create a transformer for the Earth Observation Satellite model. Child class of nn.Transformer.
    """
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            activation: str,
            batch_first: bool = True,
            kaiming_init: bool = False
        ):
        super(EOSTransformer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        self.architecture_type = "Transformer"

        if kaiming_init:
            self.override_linears_with_kaiming()

    def override_linears_with_kaiming(self):
        # Encoder
        for layer in self.encoder.layers:
            for module in layer.children():
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    init.zeros_(module.bias)

        # Decoder
        for layer in self.decoder.layers:
            for module in layer.children():
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    init.zeros_(module.bias)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
    
class Projector(nn.Module):
    """
    Class to project the output of the transformer into a deterministic or stochastic output. Child class of nn.Module.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int
        ):
        super(Projector, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        layers.append(nn.Linear(self.in_dim, self.in_dim // 2))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(self.in_dim // 2, self.out_dim))
        # layers.append(nn.LayerNorm(self.out_dim))
        self.project = nn.Sequential(*layers)

    def init_weights(self):
        for layer in self.project:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
                init.zeros_(layer.bias)

    def forward(self, x):
        return self.project(x)

#####################################################
#
# TRANSFORMER ARCHITECTURE. BOTH ENCODER AND DECODER.
#
#####################################################

class TransformerModelEOS(nn.Module):
    """
    Class to create the Earth Observation Satellite model. Child class of nn.Module. Parts:
        · 1a. embedder for the src
        · 1b. embedder for the tgt
        · 2. positional encoder
        · 3. transformer
        · 4. stochastic projector
    """
    def __init__(
            self,
            src_dim: int,
            tgt_dim: int,
            out_dim: int,
            d_model: int,
            nhead: int,
            max_len: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            embed_dropout: float,
            pos_dropout: float,
            transformer_dropout: float,
            position_encoding: str,
            activation: str = "relu",
            batch_first: bool = True,
            kaiming_init: bool = False,
            is_value_fn: bool = False,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        super(TransformerModelEOS, self).__init__()
        self.model_type = "Earth Observation Model"
        self.out_dim = out_dim
        self.is_value_fn = is_value_fn

        encoding_segment_size = int(torch.clamp(torch.tensor(d_model // 100), min=2, max=5).item()) # segment size is between 2 and 5
        embed_dim = d_model - encoding_segment_size

        src_embedder = FloatEmbedder(
            input_dim=src_dim,
            embed_dim=embed_dim if position_encoding == "segment" else d_model,
            dropout=embed_dropout
        )

        tgt_embedder = FloatEmbedder(
            input_dim=tgt_dim,
            embed_dim=embed_dim if position_encoding == "segment" else d_model,
            dropout=embed_dropout
        )

        if position_encoding == "sine":
            pos_encoder = PositionalEncoder(
                max_len=max_len,
                d_model=d_model,
                dropout=pos_dropout
            )
        elif position_encoding == "segment":
            pos_encoder = SegmentPositionalEncoder(
                max_len=max_len,
                d_model=d_model,
                encoding_segment_size=encoding_segment_size,
                dropout=pos_dropout
            )
        else:
            raise ValueError("The position encoding must be either 'sine' or 'segment'")

        transformer = EOSTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=activation,
            batch_first=batch_first,
            kaiming_init=kaiming_init
        )

        projector = Projector(
            in_dim=d_model,
            out_dim=int(2 * out_dim if not is_value_fn else out_dim)
        )

        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.pos_encoder = pos_encoder
        self.transformer = transformer
        self.projector = projector
        self.npe = NormalParamExtractor() if not is_value_fn else nn.Identity()
        self.gpu_device = device

        if self.transformer.architecture_type != "Transformer":
            raise ValueError("The model is not a transformer.")

    def forward(self, src, tgt):
        with torch.no_grad():
            # Check if src and tgt have the same number of dimensions
            if src.dim() != tgt.dim():
                raise ValueError("For cleaner code, the number of dimensions of src and tgt should be the same")
            
            # Check whether they are batched or not
            if src.dim() == 1:
                src = src.unsqueeze(0).unsqueeze(0)
                tgt = tgt.unsqueeze(0).unsqueeze(0)
            elif src.dim() == 2:
                src = src.unsqueeze(0)
                tgt = tgt.unsqueeze(0)
            
            # Check if the number of src and tgt are equal
            if src.shape[1] != tgt.shape[1]:
                raise ValueError("The number of src and tgt must be equal")
            else:
                seq_len = src.shape[1]

        # Pass the embedded src and tgt through the positional encoder
        src = self.pos_encoder(self.src_embedder(src))
        tgt = self.pos_encoder(self.tgt_embedder(tgt))

        # Set the src and tgt masks
        mask = self.transformer._generate_square_subsequent_mask(seq_len)

        # Pass the input src and tgt through the transformer
        x = self.transformer(src, tgt, src_mask=mask, tgt_mask=mask, memory_mask=mask, src_is_causal=True, tgt_is_causal=True, memory_is_causal=True)

        # Pass the output through the projector
        x = self.projector(x)

        # Apply the normal parameter extractor
        x = self.npe(x) # (loc, scale) if not is_value_fn else (value)

        # If while in training mode, return the last sequence only
        if self.training:
            if self.is_value_fn:
                return x[:, -1, :]
            else:
                loc, scale = x
                return loc[:, -1, :], scale[:, -1, :]

        return x

#################################################
#
# TRANSFORMER ENCODER ARCHITECTURE. ENCODER ONLY.
#
#################################################

class TransformerEncoderModelEOS(nn.Module):
    """
    Class to create the Earth Observation Satellite model. Child class of nn.Module. Parts:
        · 1. embedder for the src
        · 2. positional encoder
        · 3. transformer encoder
        · 4. stochastic projector
    """
    def __init__(
            self,
            src_dim: int,
            out_dim: int,
            d_model: int,
            nhead: int,
            max_len: int,
            num_encoder_layers: int,
            dim_feedforward: int,
            embed_dropout: float,
            pos_dropout: float,
            encoder_dropout: float,
            position_encoding: str,
            activation: str = "relu",
            batch_first: bool = True,
            kaiming_init: bool = False,
            is_value_fn: bool = False,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        super(TransformerEncoderModelEOS, self).__init__()
        self.model_type = "Earth Observation Model"
        self.out_dim = out_dim
        self.is_value_fn = is_value_fn

        encoding_segment_size = int(torch.clamp(torch.tensor(d_model // 100), min=2, max=5).item()) # segment size is between 2 and 5
        embed_dim = d_model - encoding_segment_size

        src_embedder = FloatEmbedder(
            input_dim=src_dim,
            embed_dim=embed_dim if position_encoding == "segment" else d_model,
            dropout=embed_dropout
        )

        if position_encoding == "sine":
            pos_encoder = PositionalEncoder(
                max_len=max_len,
                d_model=d_model,
                dropout=pos_dropout
            )
        elif position_encoding == "segment":
            pos_encoder = SegmentPositionalEncoder(
                max_len=max_len,
                d_model=d_model,
                encoding_segment_size=encoding_segment_size,
                dropout=pos_dropout
            )
        else:
            raise ValueError("The position encoding must be either 'sine' or 'segment'")

        transformer_encoder = EOSTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            activation=activation,
            batch_first=batch_first,
            kaiming_init=kaiming_init
        )

        projector = Projector(
            in_dim=d_model,
            out_dim=int(2 * out_dim if not is_value_fn else out_dim)
        )

        self.src_embedder = src_embedder
        self.pos_encoder = pos_encoder
        self.transformer_encoder = transformer_encoder
        self.projector = projector
        self.gpu_device = device
        self.npe = NormalParamExtractor() if not is_value_fn else nn.Identity()

        if self.transformer_encoder.architecture_type != "TransformerEncoder":
            raise ValueError("The model is not a transformer.")

    def forward(self, src: torch.Tensor):
        with torch.no_grad():            
            # Check whether they are batched or not
            while src.dim() < 3:
                src = src.unsqueeze(0)

        seq_len = src.shape[1]

        # Pass the embedded src through the positional encoder
        src = self.pos_encoder(self.src_embedder(src))

        # Set the src mask
        mask = self.transformer_encoder._generate_square_subsequent_mask(seq_len)

        # Pass the input src through the transformer
        x = self.transformer_encoder(src, mask=mask, is_causal=True)

        # Pass the output through the projector
        x = self.projector(x)

        # Apply the normal parameter extractor
        x: torch.Tensor = self.npe(x) # (loc, scale) if not is_value_fn else (value)

        # If while in training mode, return the last sequence only
        if self.training:
            if self.is_value_fn:
                return x[:, -1, :]
            else:
                loc, scale = x
                return loc[:, -1, :], scale[:, -1, :]

        return x

#########################################################
#
# MULTI-LAYER PERCEPTRON ARCHITECTURE. SIMPLE AND DENSE.
#
#########################################################

class MLPModelEOS(nn.Module):
    """
    Class to create a Multi-Layer Perceptron model. Child class of nn.Module.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_layers: tuple[int],
            dropout: float,
            is_value_fn: bool = False,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super(MLPModelEOS, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gpu_device = device
        self.npe = NormalParamExtractor() if not is_value_fn else nn.Identity()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_layers[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        layers.append(nn.LayerNorm(hidden_layers[-1]))
        layers.append(nn.Linear(hidden_layers[-1], 2 * out_dim if not is_value_fn else out_dim))
        self.mlp = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1).unsqueeze(1) # (batch_size, seq_len, state_dim) -> (batch_size, 1, seq_len * state_dim)

        # Check it has 2 dimensions
        if x.dim() != 3:
            raise ValueError("The input tensor must have 3 dimensions: (batch_size, seq_len, state_dim)")
        
        # Check if the input tensor has the right number of sequences
        if x.shape[-1] != self.in_dim:
            # Add zeros to the tensor where x has size (batch_size, seq_len, state_dim)
            x = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], self.in_dim - x.shape[-1]).to(self.gpu_device)], dim=-1)

        # Pass the input tensor through the MLP
        x = self.mlp(x)

        # Apply the normal parameter extractor
        x = self.npe(x) # (loc, scale) if not is_value_fn else (value)

        return x