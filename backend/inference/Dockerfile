FROM nvcr.io/nvidia/pytorch:24.11-py3 

RUN pip uninstall pynvml -y
RUN pip install nvidia-ml-py
RUN pip install llama-index-llms-vllm
RUN pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

CMD ["vllm", "serve", "allenai/OLMo-2-1124-13B-Instruct", "--dtype=half", "--tensor-parallel-size=4"]
