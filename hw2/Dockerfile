FROM python:3-alpine

RUN apk add --no-cache bash gcc musl-dev

WORKDIR /app
COPY . .

EXPOSE 8000

# Run your script
CMD ["bash", "run_demo.sh"]
