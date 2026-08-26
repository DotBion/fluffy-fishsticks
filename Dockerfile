# Serving image for the FinPulse LSTM.
#
# Built by workflows/build-container-image.yaml via Kaniko. That workflow
# clones this repo into the build context, then overwrites train/lstm_model.pth
# and train/scaler.pkl with the artifacts of the MLflow run behind the model
# version being built, so CI serves the model it just trained and a plain local
# build serves the committed one.
#
# BACKEND selects the inference path at runtime: onnx for deployment, torch for
# debugging. Both are installed so an image can be flipped without a rebuild.
FROM python:3.11-slim

WORKDIR /app

ARG MODEL_VERSION=dev
LABEL org.opencontainers.image.title="finpulse-app" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.source="https://github.com/DotBion/fluffy-fishsticks"

# INCLUDE_TORCH=false builds an ONNX-only image: a few hundred MB and about
# a minute. Set it true to also get the torch backend, which costs ~2 GB and
# a great deal of build time on ARM.
ARG INCLUDE_TORCH=false

COPY serving/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$INCLUDE_TORCH" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY serving/ ./serving/

# Weights and scaler must come from the same training run: a scaler that does
# not match the weights produces silently wrong prices rather than an error.
COPY train/scaler.pkl ./scaler.pkl
COPY App/src/lstm_model.onnx ./lstm_model.onnx
# Only useful with INCLUDE_TORCH=true; harmless otherwise and keeps the
# image able to switch backends without a rebuild once torch is present.
COPY train/lstm_model.pth ./lstm_model.pth

ENV BACKEND=onnx \
    MODEL_PATH=/app/lstm_model.pth \
    ONNX_MODEL_PATH=/app/lstm_model.onnx \
    SCALER_PATH=/app/scaler.pkl \
    MODEL_VERSION=${MODEL_VERSION} \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "serving.app:app"]
