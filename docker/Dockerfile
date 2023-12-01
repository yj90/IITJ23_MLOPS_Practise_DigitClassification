FROM python:3.9.17
COPY . /digits/
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
WORKDIR /digits
EXPOSE 5000
ENV FLASK_APP=api/app.py
CMD ["run","flask","--host=0.0.0.0"]
#CMD ["python3","digits.py","--runs=1","--test_size=0.2","--dev_size=0.3","--models=svm,tree"]