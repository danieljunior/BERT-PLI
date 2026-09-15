import time
import json
import os
from typing import Any, Dict, List, Sequence, Tuple
try:
    import pymonetdb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pymonetdb = None


class DfanalyzerService:
    URL = "dfanalyzer"
    PORT = 50000
    DATABASE = "dataflow_analyzer"
    USERNAME = "monetdb"
    PASSWORD = "monetdb"

    def __init__(self, bypass: bool = False):
        self.bypass = bypass
    
    def get_monet_connection(self):
        if self.bypass: 
            return None

        if pymonetdb is None:
            raise ModuleNotFoundError(
                "pymonetdb is required to connect to MonetDB. "
                "Install it (e.g., `pip install pymonetdb`) or run with bypass=True."
            )
        conn = pymonetdb.connect(
            hostname=self.URL,
            port=self.PORT,
            database=self.DATABASE,
            username=self.USERNAME,
            password=self.PASSWORD,
        )
        return conn

    def dataflow_exists(self, dataflow_name : str) -> bool:
        if self.bypass: 
            return False

        conn = self.get_monet_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM dataflow WHERE tag = %s;", (dataflow_name,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        return row is not None

    def get_last_task_id(self, df_tag: str) -> int:
        if self.bypass: 
            return

        conn = self.get_monet_connection()
        conn.commit()
        cursor = conn.cursor()
        query_1 = """
            SELECT t.identifier 
            FROM task t
            ORDER BY t.identifier DESC 
            LIMIT 1;
        """
        cursor.execute(query_1)
        row = cursor.fetchone()

        # If no tasks exist, return 0
        if row is None:
            last_identifier = 0
        else:
            last_identifier = row[0]

        cursor.close()
        conn.close()

        return last_identifier

    def get_last_task_id_from_dataflow(self, df_tag: str) -> int:
        if self.bypass: 
            return

        conn = self.get_monet_connection()
        cursor = conn.cursor()

        # 1. Get df_id
        cursor.execute("SELECT id FROM dataflow WHERE tag = %s;", (df_tag,))
        row = cursor.fetchone()

        if row is None:
            cursor.close()
            conn.close()
            raise ValueError(f"No dataflow found with tag='{df_tag}'")

        df_id = row[0]

        # 2. Get latest task identifier (descending order, take first)
        query_1 = """
            SELECT t.identifier 
            FROM task t
            INNER JOIN dataflow_version dv ON t.df_version = dv.version
            WHERE dv.df_id = %s
            ORDER BY dv.version DESC, t.identifier DESC 
            LIMIT 1;
        """
        cursor.execute(query_1, (df_id,))
        row = cursor.fetchone()

        # If no tasks exist, return 0
        if row is None:
            last_identifier = 0
        else:
            last_identifier = row[0]

        cursor.close()
        conn.close()

        return last_identifier

    def next_task_id(self, dataflow_tag : str) -> int:
        if self.bypass: 
            return

        last_id = self.get_last_task_id(dataflow_tag)
        return last_id + 1

    def update_custom_text_columns(self):
        conn = self.get_monet_connection()
        cursor = conn.cursor()
        already = False

        while not already:
            conn.commit()
            cursor.execute("SELECT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ds_entailment_config' AND system = false);")
            response = cursor.fetchone()[0]
            if response:
                already = True
            else:
                time.sleep(1)

        changes = [
            ['ds_entailment_config', 'config' ],
            ['ds_train_poolout_config', 'config' ],
            ['ds_test_poolout_config', 'config' ],
            ['ds_evaluate_config', 'config' ],
            ['ds_classifier_config', 'config' ],
        ]

        query = """
            ALTER TABLE %s ADD COLUMN %s varchar(5000);
            UPDATE %s SET %s = CONVERT(%s, varchar(5000));
            DROP VIEW %s restrict;
            ALTER TABLE %s DROP COLUMN %s restrict;
            ALTER TABLE %s RENAME COLUMN %s TO %s;
            CREATE VIEW %s AS SELECT * FROM %s;  
        """
        for change in changes:
            ds_table = change[0]
            column = change[1]
            tmp_column = 'tmp_'+ column
            view = ds_table.split('ds_')[1]
            cursor.execute(query % ( ds_table, tmp_column, 
                                    ds_table, tmp_column, column,
                                    view,
                                    ds_table, column,
                                    ds_table, tmp_column, column,
                                    view, ds_table,
                                )
                            )
            conn.commit()
        cursor.close()
        conn.close()

    def export_classifier_metrics(self, term: str, output_path: str = None) -> Dict[str, Any]:
        """Export classifier validation metrics stored in MonetDB.

        Filters `ds_classifier_model` rows where `checkpoint` contains `term`, loads
        per-epoch metrics from `validation_metrics_filepath` (file path or JSON), and
        returns a structure compatible with `output/results/summarized/valid_metrics.json`.
        If `output_path` is provided, dumps the JSON result to that file.

        Example:
            service.export_classifier_metrics(term="attengru", output_path="metrics.json")
        """

        self._validate_export_term(term)
        if self.bypass:
            return {"checkpoint_dir": "", "results": []}

        rows = self._fetch_classifier_model_rows(term=term)
        if not rows:
            raise ValueError(
                "No ds_classifier_model rows found for term="
                f"{term!r} (expected checkpoint LIKE %term%)"
            )

        result = self._export_classifier_metrics_from_rows(rows)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        return result

    def _validate_export_term(self, term: str) -> None:
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"term must be a non-empty str, got: {term!r}")

    def _export_classifier_metrics_from_rows(
        self, rows: Sequence[Tuple[str, int, Any]]
    ) -> Dict[str, Any]:
        chosen_by_checkpoint = self._dedupe_by_checkpoint(rows)
        results = self._build_classifier_metrics_results(chosen_by_checkpoint)
        checkpoint_dir = self._infer_checkpoint_dir([r["checkpoint"] for r in results])
        return {"checkpoint_dir": checkpoint_dir, "results": results}

    def _fetch_classifier_model_rows(self, term: str) -> List[Tuple[str, int, Any]]:
        pattern = f"%{term}%"
        conn = self.get_monet_connection()
        cursor = conn.cursor()
        try:
            conn.commit()
            query = (
                "SELECT checkpoint, epoch, validation_metrics_filepath "
                "FROM ds_classifier_model "
                "WHERE checkpoint LIKE %s "
                "ORDER BY epoch ASC;"
            )
            cursor.execute(query, (pattern,))
            rows = cursor.fetchall() or []
            return [(r[0], int(r[1]), r[2]) for r in rows]
        finally:
            cursor.close()
            conn.close()

    def _dedupe_by_checkpoint(
        self, rows: Sequence[Tuple[str, int, Any]]
    ) -> Dict[str, Tuple[int, Any]]:
        """Pick one row per checkpoint (prefer higher epoch, non-empty metrics)."""

        chosen: Dict[str, Tuple[int, Any]] = {}
        for checkpoint, epoch, metrics_ref in rows:
            if checkpoint is None:
                continue
            key = str(checkpoint)
            prev = chosen.get(key)
            if prev is None:
                chosen[key] = (epoch, metrics_ref)
                continue

            prev_epoch, prev_ref = prev
            if epoch > prev_epoch:
                chosen[key] = (epoch, metrics_ref)
                continue

            if epoch == prev_epoch and (not prev_ref) and metrics_ref:
                chosen[key] = (epoch, metrics_ref)

        return chosen

    def _build_classifier_metrics_results(
        self, chosen_by_checkpoint: Dict[str, Tuple[int, Any]]
    ) -> List[Dict[str, Any]]:
        items = sorted(chosen_by_checkpoint.items(), key=lambda kv: kv[1][0])
        results: List[Dict[str, Any]] = []
        for checkpoint, (epoch, metrics_ref) in items:
            metrics = self._load_validation_metrics(
                metrics_ref=metrics_ref, checkpoint=checkpoint, epoch=epoch
            )
            results.append(
                {
                    "checkpoint": checkpoint,
                    "epoch": epoch,
                    "metrics": metrics,
                }
            )
        return results

    def _load_validation_metrics(
        self, metrics_ref: Any, checkpoint: str, epoch: int
    ) -> Dict[str, Any]:
        text = self._normalize_metrics_ref(metrics_ref)
        data = self._read_metrics_json(text=text, source=metrics_ref)
        metrics = self._extract_metrics_dict(
            data=data, source=metrics_ref, checkpoint=checkpoint, epoch=epoch
        )
        return metrics

    def _normalize_metrics_ref(self, metrics_ref: Any) -> str:
        if metrics_ref is None:
            raise ValueError(
                "validation_metrics_filepath is None (expected a JSON string or file path)"
            )
        if isinstance(metrics_ref, bytes):
            metrics_ref = metrics_ref.decode("utf-8")
        if not isinstance(metrics_ref, str):
            raise ValueError(
                "validation_metrics_filepath must be a str/bytes (JSON or file path), "
                f"got: {type(metrics_ref).__name__}={metrics_ref!r}"
            )

        text = metrics_ref.strip()
        if not text:
            raise ValueError(
                f"validation_metrics_filepath is empty (got: {metrics_ref!r}; expected JSON or a file path)"
            )
        return text

    def _read_metrics_json(self, text: str, source: Any) -> Any:
        if text.startswith("{") or text.startswith("["):
            return json.loads(text)

        path = self._resolve_metrics_path(text=text, source=source)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_metrics_path(self, text: str, source: Any) -> str:
        path = text
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Metrics file not found: {path!r} (from validation_metrics_filepath={source!r})"
            )
        return path

    def _extract_metrics_dict(
        self, data: Any, source: str, checkpoint: str, epoch: int
    ) -> Dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
            return data["metrics"]

        if isinstance(data, dict) and isinstance(data.get("results"), list):
            metrics = self._extract_metrics_from_results_list(
                results=data["results"], source=source, checkpoint=checkpoint, epoch=epoch
            )
            if metrics is not None:
                return metrics

        if isinstance(data, dict):
            return data

        raise ValueError(
            "Validation metrics JSON must be a dict (or a dict with a 'metrics' dict); "
            f"got: {type(data).__name__} from {source!r}"
        )

    def _extract_metrics_from_results_list(
        self,
        results: Any,
        source: str,
        checkpoint: str,
        epoch: int,
    ) -> Any:
        if not isinstance(results, list):
            return None

        by_checkpoint = [r for r in results if isinstance(r, dict) and r.get("checkpoint") == checkpoint]
        for candidate in by_checkpoint:
            metrics = candidate.get("metrics")
            if isinstance(metrics, dict):
                return metrics

        by_epoch = [r for r in results if isinstance(r, dict) and r.get("epoch") == epoch]
        for candidate in by_epoch:
            metrics = candidate.get("metrics")
            if isinstance(metrics, dict):
                return metrics

        return None

    def _infer_checkpoint_dir(self, checkpoints: Sequence[str]) -> str:
        if not checkpoints:
            return ""
        if len(checkpoints) == 1:
            return os.path.dirname(checkpoints[0])

        common = os.path.commonpath(list(checkpoints))
        _, ext = os.path.splitext(common)
        if ext:
            return os.path.dirname(common)
        return common
