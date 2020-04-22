# people-counter
Uses OpenCV4 running on a local to detect faces and people. Send info to AWS S3 bucket for analysis and reporting

haarcascade files will be used for fast, efficient detection of both faces and people (because some people may be facing sideways or backwards)
Faces will periodically be uploaded to an S3 bucket in AWS. This will trigger a step function which checks each face against an existing collection using Rekognition.
If the face has been seen before, it will not be added again; but a record of the prescence of that face will be recorded in a DynamoDB table
If the face is new, it will registered in the collection and again, its prescence will be recorded in a DynamoDB table
The overall people count will be sent to AWS via an IOT topic and recorded in a time series database


This will allow for an AWS Quicksight dashboard containing such information as:
* Number of visitors at the booth over time
* Number of unique visitors over time
* Average time spent at booth per visitor
* Which visitors spent the most time at the booth
