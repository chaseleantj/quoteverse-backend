up:
	docker-compose up --build
psql:
    docker-compose exec db psql --username=postgres --dbname=quotes_db
