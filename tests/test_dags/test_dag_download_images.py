import unittest
import os
from airflow.models import DagBag


class TestMLFlowDag(unittest.TestCase):

    def setUp(self):
        dags_folder = os.getenv('DAGS_FOLDER', 'airflow/dags')
        self.dagbag = DagBag(dag_folder=dags_folder)

    def test_dag_loaded(self):
        """Test if the DAG is correctly loaded"""
        dag_id = 'mlops_project_pipeline_sqlite'
        self.assertIn(dag_id, self.dagbag.dags)
        self.assertFalse(
            len(self.dagbag.import_errors),
            f"DAG import failures: {self.dagbag.import_errors}"
        )

    def test_task_list(self):
        """Test task list in DAG"""
        dag = self.dagbag.dags['mlops_project_get_store_images']
        task_ids = list(dag.task_dict.keys())
        expected_tasks = [
            'start_task',
            'insert_urls_to_postgresql',
            'process_images',
        ]
        for task in expected_tasks:
            self.assertIn(task, task_ids)

    def test_dependencies(self):
        """Test task dependencies"""
        dag = self.dagbag.dags['mlops_project_get_store_images']
        self.assertEqual(
            dag.task_dict['insert_urls_to_postgresql'].upstream_task_ids,
            {'start_task'}
        )
        self.assertEqual(
            dag.task_dict['process_images'].upstream_task_ids,
            {'insert_urls_to_postgresql'}
        )


if __name__ == '__main__':
    unittest.main()
