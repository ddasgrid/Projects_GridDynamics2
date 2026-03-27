graph TD
    %% Define Styles
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef hardware fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;

    %% Phase 1: The Split
    A([Trained PyTorch Model]) ::: input --> B{Architecture Split}
    B -->|Extract Vision| C(Vision Transformer) ::: process
    B -->|Extract Logic| D(BERT + Fusion + SwiGLU) ::: process

    %% Phase 2: Vision Quantization
    C --> E[torch.onnx.export] ::: process
    E --> F[FP32 ONNX Model] ::: model
    F --> G[Dynamic INT8 Quantization] ::: process
    G --> H[INT8 ONNX Model] ::: model

    %% Phase 3: CoreML Export
    D --> I[torch.jit.trace] ::: process
    I --> J[coremltools.convert] ::: process
    J --> K[CoreML .mlpackage] ::: model

    %% Phase 4: Hybrid Inference (Deployment)
    subgraph Hybrid Edge Inference Pipeline
        L(Raw Image) ::: input --> M[ONNX CPU Provider] ::: hardware
        H -.-> M
        M --> O(Visual Features)

        N(Raw Text) ::: input --> P[PyTorch MPS / GPU] ::: hardware
        N -.-> P
        P --> Q(Text Features)

        O --> R[CoreML Neural Engine] ::: hardware
        Q --> R
        K -.-> R
        
        R --> S([Final Entailment Prediction]) ::: input
    end
