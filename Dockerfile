FROM tensorflow/tensorflow:1.13.1-py3

COPY . /
RUN chmod 755 start.sh

RUN apt-get update && apt-get install --no-install-recommends -y libglib2.0-0 && apt-get clean

RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install /cb-core/

EXPOSE 80

ENTRYPOINT [ "/start.sh" ]