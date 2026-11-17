"""Evaluate matched shoulder/elbow/wrist estimates against independent 3D reference.

The result applies to the recorded conditions, not every camera pose or robot task.
This tool never enables physical actuation or treats model confidence as accuracy.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

POINTS=('shoulder','elbow','wrist')
COLUMNS=['point','estimated_timestamp_ms','reference_timestamp_ms','estimated_x_m','estimated_y_m','estimated_z_m','reference_x_m','reference_y_m','reference_z_m']


def evaluate(rows,max_error_mm=60.,min_per_point=10,max_skew_ms=20.):
    grouped={name:[] for name in POINTS}
    for row in rows:
        if row['point'] not in grouped:
            raise ValueError('Point must be shoulder, elbow or wrist')
        values=np.array([float(row[key]) for key in COLUMNS[1:]])
        if not np.isfinite(values).all():
            raise ValueError('Reference/estimate values must be finite')
        if abs(values[0]-values[1])>max_skew_ms:
            raise ValueError('Reference/estimate timestamps are not aligned within the configured tolerance')
        grouped[row['point']].append(float(np.linalg.norm(values[2:5]-values[5:8])*1000))
    statistics={}
    for point,errors in grouped.items():
        if len(errors)<min_per_point:
            raise ValueError(f'Need at least {min_per_point} independent matched samples for {point}')
        statistics[point]={'samples':len(errors),'mean_mm':float(np.mean(errors)),
                           'p95_mm':float(np.percentile(errors,95)),'max_mm':float(np.max(errors))}
    passed=all(item['max_mm']<=max_error_mm for item in statistics.values())
    return {'passed':passed,'accuracy_status':'passed_for_recorded_conditions' if passed else 'failed',
            'criterion':'maximum 3D Euclidean error for each tracked arm point',
            'max_error_threshold_mm':max_error_mm,'max_timestamp_skew_ms':max_skew_ms,
            'minimum_samples_per_point':min_per_point,'statistics':statistics,
            'physical_actuation_enabled':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-description',required=True,help='Describe the independent measuring system and alignment/calibration')
    parser.add_argument('--max-error-mm',type=float,default=60.)
    parser.add_argument('--min-per-point',type=int,default=10)
    parser.add_argument('--max-skew-ms',type=float,default=20.)
    args=parser.parse_args()
    if not np.isfinite([args.max_error_mm,args.max_skew_ms]).all() or args.max_error_mm<=0 or args.max_skew_ms<0 or args.min_per_point<1:
        parser.error('Invalid evaluation thresholds')
    with args.csv.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    try:
        report=evaluate(rows,args.max_error_mm,args.min_per_point,args.max_skew_ms)
    except (ValueError,KeyError) as error:
        parser.error(str(error))
    report.update({'reference_description':args.reference_description,'created_utc':datetime.now(timezone.utc).isoformat(),
                   'input_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as output:
        json.dump(report,output,indent=2,allow_nan=False)
    print('PASS for recorded conditions' if report['passed'] else 'FAIL: arm tracking exceeds the configured position-error threshold')
    for point,item in report['statistics'].items():
        print(f"{point}: mean {item['mean_mm']:.1f}, p95 {item['p95_mm']:.1f}, max {item['max_mm']:.1f} mm")
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
