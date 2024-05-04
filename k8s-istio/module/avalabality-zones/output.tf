output "aws-azs" {
  value = data.aws_availability_zones.current.names
}