# Gunakan image resmi TensorFlow Serving
FROM tensorflow/serving:latest

# Salin model ke lokasi yang sesuai di dalam container
COPY ./serving_model_dir /models/spam-prediction-model
COPY ./config /model_config

# Setel variabel lingkungan untuk nama model dan path dasar model
ENV MODEL_NAME spam-prediction-model
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh