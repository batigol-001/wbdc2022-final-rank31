FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-1.0.0

WORKDIR /opt/ml/wxcode

ADD . /opt/ml/wxcode


RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple


# RUN pip install nvidia-tensorrt -U --index-url https://pypi.ngc.nvidia.com
# RUN unzip torch2trt.zip

# RUN python torch2trt/setup.py install



CMD sh -c "sh start.sh"
