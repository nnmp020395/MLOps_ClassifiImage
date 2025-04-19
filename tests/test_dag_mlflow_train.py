import unittest
from airflow.models import DagBag

class TestMLflowTrainDag(unittest.TestCase):

    def setUp(self):
        self.dagbag = DagBag()

    def test_dag_loaded(self):
        """Le DAG doit être correctement chargé"""
        dag_id = "dinov2_train_pipeline_with_mlflow"
        self.assertIn(dag_id, self.dagbag.dags)
        self.assertFalse(
            len(self.dagbag.import_errors),
            f"Import errors dans DagBag: {self.dagbag.import_errors}"
        )

    def test_task_exists(self):
        """La tâche 'mlflow_tracking' doit exister dans le DAG"""
        dag = self.dagbag.dags["dinov2_train_pipeline_with_mlflow"]
        task_ids = list(dag.task_dict.keys())
        self.assertIn("mlflow_tracking", task_ids)

    def test_task_is_python_operator(self):
        """La tâche doit être un PythonOperator"""
        dag = self.dagbag.dags["dinov2_train_pipeline_with_mlflow"]
        task = dag.get_task("mlflow_tracking")
        from airflow.operators.python import PythonOperator
        self.assertIsInstance(task, PythonOperator)


if __name__ == '__main__':
    unittest.main()
