import requests
import time
from pathlib import Path
import json
from tqdm import tqdm


class DfanalyzerService:
    HOST = "dfanalyzer"
    PORT = 22000
    HEADERS = {"Content-type": "application/json"}

    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port

    def create_dataflow(self, json_path: str):
        """
        Cria um dataflow no servidor local a partir de um arquivo JSON.

        Args:
            json_path: Caminho para o arquivo JSON do dataflow
        """
        df = json.dumps(json.loads(Path(json_path).read_text()))

        try:
            requests.post(
                f"http://{self.host}:{self.port}/pde/dataflow/json",
                data=df,
                headers=self.HEADERS,
            )
        except Exception as ex:
            print(f"Erro ao enviar dataflow: {ex}")

    def register_task(self, json_path: str):
        df = json.dumps(json.loads(Path(json_path).read_text()))

        try:
            requests.post(
                f"http://{self.host}:{self.port}/pde/task/json",
                data=df,
                headers=self.HEADERS,
            )
        except Exception as ex:
            print(f"Erro ao enviar tarefa: {ex}")

    def e(self, dt, num_msg, num_elements):
        """
        Envia mensagens com elementos de dados para um servidor local.

        Args:
            dt: Delay em segundos entre mensagens
            num_msg: Número de mensagens a enviar
            num_elements: Número total de elementos
        """
        num_elements_per_msg = num_elements // num_msg
        output_file = f"./dt{dt}_num_msg{num_msg}_num_el{num_elements}.data"

        print(
            f"Sending {num_elements} data elements distributed in {num_msg} with {num_elements_per_msg} element(s) per message"
        )

        # Dataflow as a Python dict
        df_dict = {
            "tag": "uq_rtm",
            "transformations": [
                {
                    "programs": [
                        {"name": "SparseGrid", "path": "/bin/sparse_grid_cc_dataset"}
                    ],
                    "tag": "sparse_grid_construction",
                    "sets": [
                        {
                            "tag": "sparsegridinput",
                            "attributes": [
                                {"name": "vmid", "type": "NUMERIC"},
                                {"name": "dimension", "type": "NUMERIC"},
                                {"name": "level", "type": "NUMERIC"},
                            ],
                            "type": "INPUT",
                        },
                        {
                            "tag": "sparsegridoutput",
                            "attributes": [
                                {"name": "vmid", "type": "NUMERIC"},
                                {"name": "dimension", "type": "NUMERIC"},
                                {"name": "level", "type": "NUMERIC"},
                                {"name": "region", "type": "FILE"},
                                {"name": "weights", "type": "FILE"},
                                {"name": "points", "type": "FILE"},
                            ],
                            "type": "OUTPUT",
                        },
                    ],
                }
            ],
        }

        # Serialize to JSON (the rest of the code expects `df` as a JSON string)
        # df = json.dumps(df_dict)
        df = json.loads(
            "" + Path("./provenance/dataflow_example.json").read_text() + ""
        )
        df = json.dumps(df)
        try:
            requests.post(
                f"http://{self.host}:{self.port}/pde/dataflow/json",
                data=df,
                headers=self.HEADERS,
            )
        except Exception as ex:
            print(f"Erro ao enviar dataflow: {ex}")

        # Construir elementos
        # elements = ",".join(
        #     [f'"1;8;{j};r;w;p"' for j in range(1, num_elements_per_msg + 1)]
        # )
        elements_list = [f"1;8;{j};r;w;p" for j in range(1, num_elements_per_msg + 1)]

        # Enviar mensagens
        for i in tqdm(range(1, num_msg + 1), leave=True):

            content_dict = {
                "subid": str(i),
                "workspace": "/home/luciano/Desktop/pg",
                "dataflow": "uq_rtm",
                "sets": [
                    {"elements": ["1;8;1"], "tag": "sparsegridinput"},
                    {"elements": elements_list, "tag": "sparsegridoutput"},
                ],
                "dependency": {},
                "resource": "local",
                "id": str(i),
                "transformation": "sparse_grid_construction",
                "status": "FINISHED",
            }

            content = json.dumps(content_dict)
            # content = f'{{"subid":"{i}","workspace":"/home/luciano/Desktop/pg","dataflow":"uq_rtm","sets":[{{"elements":["1;8;1"],"tag":"sparsegridinput"}}, {{"elements":[{elements}],"tag":"sparsegridoutput"}}],"dependency":{{}},"resource":"local","id":"{i}","transformation":"sparse_grid_construction","status":"FINISHED"}}'

            try:
                requests.post(
                    f"http://{self.host}:{self.port}/pde/task/json",
                    data=content,
                    headers=self.HEADERS,
                )
            except Exception as ex:
                print(f"Erro ao enviar tarefa {i}: {ex}")

            print("Sent message ", i)
            time.sleep(dt)


if __name__ == "__main__":
    # dt = 0.100  # sec
    # num_msg = 50
    # num_elements = num_msg
    dfanalyzer = DfanalyzerService()
    # dfanalyzer.e(dt, num_msg, num_elements)
    # dfanalyzer.create_dataflow("./provenance/bert_pli_dataflow.json")
    dfanalyzer.register_task("./provenance/bert_pli_restrospective_provenance.json")