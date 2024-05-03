data "aws_availability_zones" "current" {
  state = "available"
  filter {
    name   = "region-name"
    values = [var.aws-region]
  }
}