import pandas as pd
import pytest

from conjunctions.schema import EXPECTED_COLUMNS


@pytest.fixture
def messages():
    rows = []
    for event, mission, final_risk in [(1, 3, -4.), (2, 4, -30.), (3, 5, -5.), (4, 6, -12.)]:
        for t, risk in [(4., -8.), (2., -7.), (.5, final_risk)]:
            row = {c: 0. for c in EXPECTED_COLUMNS}
            row.update(event_id=event, mission_id=mission, time_to_tca=t, risk=risk,
                       miss_distance=1000. + 100 * t, c_object_type="UNKNOWN")
            rows.append(row)
    return pd.DataFrame(rows)
