FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-1.0.0

WORKDIR /opt/ml/wxcode

ADD . /opt/ml/wxcode


RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

CMD sh -c "sh start.sh"
