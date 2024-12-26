up:
	docker-compose up --build
db:
    docker-compose exec db psql --username=postgres --dbname=quotes_db
