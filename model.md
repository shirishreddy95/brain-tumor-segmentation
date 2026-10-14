 INPUT: 4 MRI Modalities (FLAIR, T1, T1ce, T2)
┌─────────────────────────────────────────────┐
│     INPUT LAYER: 3,538,944 nodes           │
│           [4 × 96³] Input Volume            │
└─────────────────┬───────────────────────────┘
                 │
                 ▼
       ┌─────────────────┐
       │   ENCODER PATH  │ (Compression - 8 Layers)
       └─────────────────┘
                 │
   ┌─────────────▼─────────────┐
   │ Enc1: 4→32 channels       │ ◄─┐ Layer 1-2: Conv3d + BatchNorm
   │ Size: 96³ = 28.3M nodes   │   │ Parameters: 3,488 + 64
   └─────────────┬─────────────┘   │
                 │ MaxPool(2)      │ Skip
   ┌─────────────▼─────────────┐   │ Connections
   │ Enc2: 32→64 channels      │ ◄─┤ Layer 3-4: Conv3d + BatchNorm
   │ Size: 48³ = 7.1M nodes    │   │ Parameters: 55,360 + 128
   └─────────────┬─────────────┘   │
                 │ MaxPool(2)      │
   ┌─────────────▼─────────────┐   │
   │ Enc3: 64→128 channels     │ ◄─┤ Layer 5-6: Conv3d + BatchNorm
   │ Size: 24³ = 1.8M nodes    │   │ Parameters: 221,312 + 256
   └─────────────┬─────────────┘   │
                 │ MaxPool(2)      │
   ┌─────────────▼─────────────┐   │
   │ Enc4: 128→256 channels    │ ◄─┤ Layer 7-8: Conv3d + BatchNorm
   │ Size: 12³ = 442K nodes    │   │ Parameters: 884,992 + 512
   └─────────────┬─────────────┘   │
                 │ MaxPool(2)      │
       ┌─────────▼─────────┐       │
       │    BOTTLENECK     │       │ Layer 9-10: Conv3d + BatchNorm
       │ 256→512 channels  │       │ Parameters: 3,539,456 + 1,024
       │ Size: 6³ = 111K nodes     │ (Most compressed)
       └─────────┬─────────┘       │
                 │ Upsample        │
       ┌─────────▼─────────┐       │
       │   DECODER PATH    │       │ (Reconstruction - 8 Layers)
       └───────────────────┘       │
                 │                 │
   ┌─────────────▼─────────────┐   │
   │ Dec1: (512+256)→256 ch    │ ◄─┘ Layer 11-12: Conv3d + BatchNorm
   │ Size: 12³ = 442K nodes    │     Parameters: 5,308,672 + 512
   └─────────────┬─────────────┘
                 │ Upsample
   ┌─────────────▼─────────────┐
   │ Dec2: (256+128)→128 ch    │ ◄─┘ Layer 13-14: Conv3d + BatchNorm
   │ Size: 24³ = 1.8M nodes    │     Parameters: 1,327,232 + 256
   └─────────────┬─────────────┘
                 │ Upsample
   ┌─────────────▼─────────────┐
   │ Dec3: (128+64)→64 ch      │ ◄─┘ Layer 15-16: Conv3d + BatchNorm
   │ Size: 48³ = 7.1M nodes    │     Parameters: 331,840 + 128
   └─────────────┬─────────────┘
                 │ Upsample
   ┌─────────────▼─────────────┐
   │ Dec4: (64+32)→32 ch       │ ◄─┘ Layer 17-18: Conv3d + BatchNorm
   │ Size: 96³ = 28.3M nodes   │     Parameters: 82,976 + 64
   └─────────────┬─────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│        OUTPUT LAYER: 3,538,944 nodes       │ Layer 19: Conv3d (1×1×1)
│   [4 × 96³] Segmentation Map               │ Parameters: 132
│ 0=Background, 1=Necrotic, 2=Edema, 3=Tumor │
└─────────────────────────────────────────────┘