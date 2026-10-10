import boto3

def setup_s3_buckets():
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )
    for bucket in ["dvc-storage", "mlflow-artifacts"]:
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"Bucket '{bucket}' existe déjà.")
        except Exception:
            s3.create_bucket(Bucket=bucket)
            print(f"Bucket '{bucket}' créé avec succès.")

if __name__ == "__main__":
    setup_s3_buckets()