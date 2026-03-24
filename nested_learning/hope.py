"""
HOPE — The Full Nested Learning Architecture
==============================================

PLAIN ENGLISH:
    HOPE is the proof-of-concept architecture that puts all the Nested Learning
    ideas together into one model. It's a sequence model (like a Transformer)
    that can:

    1. READ from a learned memory (associative memory / attention)
    2. WRITE to that memory using surprise-based updates
    3. MODIFY ITS OWN WEIGHTS during the forward pass (self-modification)
    4. OPERATE AT MULTIPLE TIME SCALES (continuum memory)

    Think of it like this:
    - A Transformer is a student with a fixed set of notes (weights) and a
      whiteboard (attention context window).
    - HOPE is a student who can rewrite their notes AS they study, has
      multiple notebooks for different time horizons, and can even change
      how they take notes based on what they're learning.

HOW THE BLOCKS FIT TOGETHER:
    Each HOPE block contains:

    ┌─────────────────────────────────────────┐
    │              HOPEBlock                   │
    │                                          │
    │  1. Self-Modifying Attention             │
    │     (attention that rewrites its weights)│
    │                                          │
    │  2. Associative Memory Gate              │
    │     (surprise-based memory read/write)   │
    │                                          │
    │  3. Continuum Memory System              │
    │     (multi-frequency MLP chain)          │
    │                                          │
    │  4. Feed-Forward Network                 │
    │     (standard MLP for mixing features)   │
    └─────────────────────────────────────────┘

    Stack N of these blocks and you get HOPE.

NESTED OPTIMIZATION LEVELS:
    Level 0 (fastest):  Self-modification of attention weights — runs every token
    Level 1:            Associative memory updates — runs every token with momentum
    Level 2:            CMS level 0 — runs every step
    Level 3:            CMS level 1 — runs every C_1 steps
    ...
    Level L:            CMS level L-2 — runs every C_{L-2} steps
    Level L+1 (slowest): Backpropagation — runs every batch

    Each level has its own learning rate, its own information flow, and its
    own update frequency. This IS the nested optimization.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, List, Optional, Tuple

from nested_learning.associative_memory import AssociativeMemory
from nested_learning.self_modifying import SelfModifyingLinear
from nested_learning.continuum_memory import ContinuumMemorySystem


class SelfModifyingAttention(nn.Module):
    """Multi-head attention where projection weights self-modify.

    This is standard multi-head attention BUT the Q, K, V projection layers
    are SelfModifyingLinear — they adjust their weights based on each input.

    The attention mechanism itself is standard softmax attention.
    The self-modification happens in the linear projections around it.

    Parameters:
        dim:       Model dimension
        num_heads: Number of attention heads
        mod_lr:    Self-modification learning rate
    """
    dim: int
    num_heads: int = 4
    mod_lr: float = 0.01

    @nn.compact
    def __call__(self, x, fast_weights=None, mask=None):
        """
        Args:
            x: Input, shape (batch, seq, dim)
            fast_weights: Dict of fast weights for Q/K/V projections
            mask: Optional causal mask

        Returns:
            output: Attention output, shape (batch, seq, dim)
            new_fast_weights: Updated fast weights
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.dim // self.num_heads

        # Self-modifying Q, K, V projections
        fw_q = fast_weights["q"] if fast_weights else None
        fw_k = fast_weights["k"] if fast_weights else None
        fw_v = fast_weights["v"] if fast_weights else None

        q_proj = SelfModifyingLinear(self.dim, mod_lr=self.mod_lr, name="q_proj")
        k_proj = SelfModifyingLinear(self.dim, mod_lr=self.mod_lr, name="k_proj")
        v_proj = SelfModifyingLinear(self.dim, mod_lr=self.mod_lr, name="v_proj")

        queries, new_fw_q = q_proj(x, fw_q)
        keys, new_fw_k = k_proj(x, fw_k)
        values, new_fw_v = v_proj(x, fw_v)

        # Reshape for multi-head attention
        queries = queries.reshape(batch_size, seq_len, self.num_heads, head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads, head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        queries = jnp.transpose(queries, (0, 2, 1, 3))
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = jnp.transpose(values, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn_weights = jnp.matmul(queries, keys.transpose(0, 1, 3, 2)) / scale

        # Causal mask (so position i can only attend to positions <= i)
        if mask is None:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_weights = jnp.where(mask[None, None, :, :] == 0, -1e9, attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Weighted sum of values
        attn_output = jnp.matmul(attn_weights, values)

        # Reshape back: (batch, heads, seq, hd) -> (batch, seq, dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim)

        # Output projection
        output = nn.Dense(self.dim, name="out_proj")(attn_output)

        new_fast_weights = {"q": new_fw_q, "k": new_fw_k, "v": new_fw_v}
        return output, new_fast_weights


class HOPEBlock(nn.Module):
    """One block of the HOPE architecture.

    Combines all four components of Nested Learning:
    1. Self-modifying attention (Level 0 — fastest optimization)
    2. Associative memory gate (Level 1 — surprise-based)
    3. Continuum memory system (Levels 2..L — multi-frequency)
    4. Feed-forward network (standard feature mixing)

    Each component has its own update rate, creating a nested hierarchy.

    Parameters:
        dim:            Model dimension
        num_heads:      Number of attention heads
        ffn_dim:        Feed-forward hidden dimension
        memory_dim:     Associative memory internal dimension
        cms_num_levels: Number of CMS levels (= number of additional optimization levels)
        cms_chunk_sizes: Update frequencies for CMS levels
        mod_lr:         Self-modification learning rate
        memory_lr:      Associative memory learning rate
        dropout_rate:   Dropout rate
    """
    dim: int
    num_heads: int = 4
    ffn_dim: int = 512
    memory_dim: int = 64
    cms_num_levels: int = 3
    cms_chunk_sizes: Optional[List[int]] = None
    mod_lr: float = 0.01
    memory_lr: float = 0.01
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x,
        memory_state=None,
        fast_weights=None,
        step: int = 0,
        training: bool = True,
    ):
        """
        Args:
            x:             Input, shape (batch, seq, dim)
            memory_state:  State from associative memory (carries across chunks)
            fast_weights:  Self-modification state (carries across chunks)
            step:          Current training step (for CMS update scheduling)
            training:      Whether in training mode

        Returns:
            output:            Block output, same shape as x
            new_memory_state:  Updated associative memory state
            new_fast_weights:  Updated self-modification weights
            cms_level_outputs: Per-level CMS outputs (for analysis)
        """
        residual = x

        # 1. SELF-MODIFYING ATTENTION
        #    The Q/K/V projections adapt their weights as data flows through
        x_norm = nn.LayerNorm(name="attn_norm")(x)
        attn_out, new_fast_weights = SelfModifyingAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            mod_lr=self.mod_lr,
            name="self_mod_attn",
        )(x_norm, fast_weights)

        if training:
            attn_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_out)
        x = residual + attn_out

        # 2. ASSOCIATIVE MEMORY GATE
        #    Read from and write to the neural memory based on surprise
        residual = x
        x_norm = nn.LayerNorm(name="memory_norm")(x)
        memory_out, new_memory_state = AssociativeMemory(
            dim=self.dim,
            memory_dim=self.memory_dim,
            num_heads=self.num_heads,
            lr=self.memory_lr,
            name="assoc_memory",
        )(x_norm, memory_state)

        # Gate: let the model learn how much to use memory vs. pass-through
        memory_gate = nn.Dense(self.dim, name="memory_gate")(x_norm)
        memory_gate = jax.nn.sigmoid(memory_gate)
        x = residual + memory_gate * memory_out

        # 3. CONTINUUM MEMORY SYSTEM
        #    Multi-frequency MLP chain — different levels update at different rates
        residual = x
        x_norm = nn.LayerNorm(name="cms_norm")(x)
        cms_out, cms_level_outputs = ContinuumMemorySystem(
            dim=self.dim,
            hidden_dim=self.ffn_dim,
            num_levels=self.cms_num_levels,
            chunk_sizes=self.cms_chunk_sizes,
            name="cms",
        )(x_norm, step=step, training=training)
        x = residual + cms_out

        # 4. FEED-FORWARD NETWORK
        #    Standard feature mixing (same as in a Transformer block)
        residual = x
        x_norm = nn.LayerNorm(name="ffn_norm")(x)
        ffn_out = nn.Dense(self.ffn_dim, name="ffn_up")(x_norm)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dense(self.dim, name="ffn_down")(ffn_out)
        if training:
            ffn_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ffn_out)
        x = residual + ffn_out

        return x, new_memory_state, new_fast_weights, cms_level_outputs


class HOPE(nn.Module):
    """HOPE: Hierarchical Optimization with Persistent Embeddings.

    The full Nested Learning sequence model. Stacks multiple HOPEBlocks,
    each containing self-modifying attention, associative memory,
    continuum memory, and feed-forward layers.

    This model can:
    - Process sequences (like a Transformer)
    - Adapt its weights during inference (self-modification)
    - Maintain long-term memory across chunks (associative memory)
    - Operate at multiple time scales (continuum memory)

    Parameters:
        vocab_size:     Size of the token vocabulary
        dim:            Model dimension
        num_layers:     Number of HOPE blocks to stack
        num_heads:      Number of attention heads per block
        ffn_dim:        Feed-forward hidden dimension
        max_seq_len:    Maximum sequence length
        memory_dim:     Associative memory internal dimension
        cms_num_levels: Number of CMS levels per block
        cms_chunk_sizes: CMS update frequencies
        mod_lr:         Self-modification learning rate
        memory_lr:      Associative memory learning rate
        dropout_rate:   Dropout rate
    """
    vocab_size: int
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    max_seq_len: int = 1024
    memory_dim: int = 64
    cms_num_levels: int = 3
    cms_chunk_sizes: Optional[List[int]] = None
    mod_lr: float = 0.01
    memory_lr: float = 0.01
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        input_ids,
        memory_states=None,
        fast_weights_list=None,
        step: int = 0,
        training: bool = True,
    ):
        """
        Args:
            input_ids:          Token IDs, shape (batch, seq_len)
            memory_states:      List of memory states, one per layer
            fast_weights_list:  List of fast weights, one per layer
            step:               Current training step
            training:           Whether in training mode

        Returns:
            logits:                 Token predictions, shape (batch, seq_len, vocab_size)
            new_memory_states:      Updated memory states (pass to next chunk)
            new_fast_weights_list:  Updated fast weights (pass to next chunk)
            all_cms_outputs:        CMS level outputs from all layers (for analysis)
        """
        batch_size, seq_len = input_ids.shape

        # Token + position embeddings
        token_emb = nn.Embed(self.vocab_size, self.dim, name="token_embed")(input_ids)
        pos_emb = nn.Embed(self.max_seq_len, self.dim, name="pos_embed")(
            jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        )
        x = token_emb + pos_emb

        if training:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Initialize states if not provided
        if memory_states is None:
            memory_states = [None] * self.num_layers
        if fast_weights_list is None:
            fast_weights_list = [None] * self.num_layers

        new_memory_states = []
        new_fast_weights_list = []
        all_cms_outputs = []

        # Process through each HOPE block
        for i in range(self.num_layers):
            x, mem_state, fast_w, cms_outs = HOPEBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                memory_dim=self.memory_dim,
                cms_num_levels=self.cms_num_levels,
                cms_chunk_sizes=self.cms_chunk_sizes,
                mod_lr=self.mod_lr,
                memory_lr=self.memory_lr,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )(x, memory_states[i], fast_weights_list[i], step, training)

            new_memory_states.append(mem_state)
            new_fast_weights_list.append(fast_w)
            all_cms_outputs.append(cms_outs)

        # Final layer norm + linear head to predict tokens
        x = nn.LayerNorm(name="final_norm")(x)
        logits = nn.Dense(self.vocab_size, name="lm_head")(x)

        return logits, new_memory_states, new_fast_weights_list, all_cms_outputs
