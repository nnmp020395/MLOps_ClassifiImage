import pytest
from airflow.models import DagBag

@pytest.fixture
def dagbag():
    return DagBag(include_examples=False)

def test_dag_integrity(dagbag):
    assert not dagbag.import_errors

    for dag_id, dag in dagbag.dags.items():
        # Check if DAG has any structural issues
        dag.test()