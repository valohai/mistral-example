# GPU Environments

## Dependencies

Resolve and lock dependencies on GPU environments:

```bash
uv pip compile requirements.in -o requirements-gpu.txt
```

## Docker Image

Build the GPU enabled Docker image:

```bash
docker build -f Dockerfile.gpu -t llm-toolkit:dev-gpu .
```

Smoke test the Docker image:

```bash
docker run -it --rm -v $(pwd):/workspace llm-toolkit:dev-gpu /bin/bash
python -c "import torch; print(torch.__version__)"
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

Release a new version of the GPU enabled Docker image:

```bash
export LLM_TOOLKIT_VERSION=0.2-gpu
docker tag llm-toolkit:dev-gpu valohai/llm-toolkit:$LLM_TOOLKIT_VERSION
docker push valohai/llm-toolkit:$LLM_TOOLKIT_VERSION
```

Cleanup:

```bash
docker rmi valohai/llm-toolkit:$LLM_TOOLKIT_VERSION
docker rmi llm-toolkit:dev-gpu
```
