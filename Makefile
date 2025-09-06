# Makefile para facilitar execução do projeto TORCS Deep RL Simulator

.PHONY: build up down train play web clean

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

train:
	docker-compose run --rm app python training/train.py --config training/configs/ppo.yaml

play:
	docker-compose run --rm app python examples/demo_play.sh

web:
	docker-compose run --rm -p 8000:8000 app uvicorn web.app:app --host 0.0.0.0 --port 8000

clean:
	docker system prune -af --volumes
