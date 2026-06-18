Se já tiver construído a imagem e só quiser subir: 
docker compose up 

OBS: sempre que der Docker compose down e depois Docker compose up, reseta o banco de dados.


# DfAnalyzer com Docker

**PASSO 1:** baixar todos os arquivos da pasta https://drive.google.com/drive/folders/1Z6MrGZT8KbGDjGxb4ampzg6Z4becLwD5?usp=sharing

## 🛠️ PASSO 2: Build da imagem Docker

Abra o terminal na pasta do projeto e rode:

```bash
docker compose down --volumes --remove-orphans
docker build --tag dfanalyzer .

OU

docker build --no-cache --tag dfanalyzer .


**PASSO 3:** subir o container com docker-compose

```bash
docker compose up dfanalyzer
```

**PASSO 4 _(opcional)_:** para os clientes (caso distribuído), lembrar de configurar a variável de ambiente DFA_URL no docker-compose.yaml

```	
	[...]
	client:
		[...]
		environment:
			- DFA_URL=http://localhost:22000
		[...]
```

--------------------------------------------
#Acessar o cluster

**listar containeres**
docker ps

**Acessar o container**
docker exec -it dfanalyzer-docker-dfanalyzer-1 bash


**Acessar o monetdb dentro do cluster**
mclient -u monetdb -d dataflow_analyzer

SELECT * FROM attribute;
SELECT * FROM data_dependency;
SELECT * FROM data_set;
SELECT * FROM data_transformation;
SELECT * FROM dataflow;
SELECT * FROM dataflow_version;
SELECT * FROM ds_idownloadvideos;
SELECT * FROM ds_odownloadvideos;
SELECT * FROM extractor;
SELECT * FROM extractor_combination;
SELECT * FROM file;
SELECT * FROM performance;
SELECT * FROM program;
SELECT * FROM task;
SELECT * FROM use_program;
SELECT * FROM ds_script_metadata;
SELECT * FROM ds_system_metadata;