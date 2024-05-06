variable "env_name" {
  description = "Name of environment"
  default     = "Development"
}

variable "aws_region" {
  description = "AWS region to provision"
  default     = "ap-southeast-1"
}


variable "bucket_name" {
  description = "s3 bucket name"
  default     = "smartminingsolutionartifactsbucket"
}