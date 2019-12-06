from datetime import datetime
import boto3
import pandas as pd

PERIOD = 300
NAME = "InstanceId"
INSTANCE_ID = "i-0369117365f36752b"

cloud_watch = boto3.client('cloudwatch', 'us-east-2')


def get_metric():
    print('Getting data from Cloudwatch')

    response = cloud_watch.get_metric_data(
        MetricDataQueries=[
        {
          "Id": "m1",
          "MetricStat": {
            "Metric": {
                "Namespace": "AWS/EC2",
                "Dimensions": [
                    {
                        "Name": NAME,
                        "Value": INSTANCE_ID
                    }
                ],
                "MetricName": "NetworkIn"
            },

            "Period": PERIOD,
            "Stat": "Maximum"
          },

          "ReturnData": True
        },
        {
            "Id": "m2",
            "MetricStat": {
              "Metric": {
                  "Namespace": "AWS/EC2",
                  "Dimensions": [
                      {
                          "Name": NAME,
                          "Value": INSTANCE_ID
                      }
                  ],
                  "MetricName": "NetworkOut"
              },

              "Period": PERIOD,
              "Stat": "Maximum"
            },

            "ReturnData": True
          },
          {
            "Id": "m3",
            "MetricStat": {
              "Metric": {
                  "Namespace": "System/Linux",
                  "Dimensions": [
                      {
                          "Name": NAME,
                          "Value": INSTANCE_ID
                      }
                  ],
                  "MetricName": "MemoryUtilization"
              },

              "Period": PERIOD,
              "Stat": "Maximum"
            },

            "ReturnData": True
          },
          {
            "Id": "m4",
            "MetricStat": {
              "Metric": {
                  "Namespace": "AWS/EC2",
                  "Dimensions": [
                      {
                          "Name": NAME,
                          "Value": INSTANCE_ID
                      }
                  ],
                  "MetricName": "CPUUtilization"
              },

              "Period": PERIOD,
              "Stat": "Maximum"
            },

            "ReturnData": True
          }
        ],
        StartTime=datetime(2019, 9, 15),
        # EndTime=datetime(2019, 10, 30),
        EndTime=datetime.now(),
        # NextToken='string',
        ScanBy='TimestampAscending'
    )

    df = pd.DataFrame()
    df['Timestamps'] = response['MetricDataResults'][0]['Timestamps']

    frame_len = len(response['MetricDataResults'][0]['Timestamps'])  # Problem with less length memory values
    memory_len = len(response['MetricDataResults'][2]['Timestamps'])
    if frame_len != memory_len:
        response['MetricDataResults'][2]['Values'] = \
            [0 for i in range(frame_len - memory_len)] + response['MetricDataResults'][2]['Values']

    for i in range(4):
        df[response['MetricDataResults'][i]['Label']] = response['MetricDataResults'][i]['Values']

    df.to_csv('data.csv', index=False)
