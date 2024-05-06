variable "aws-region" {
  description = "Enter the aws region to work"
  default     = "ap-southeast-1"
}

variable "key_name" {
  default = "ksv-singapur"
}


variable "public-ip-cidr" {
  default = ["10.0.0.0/24", "10.0.1.0/24", "10.0.2.0/24"]
}

variable "private-ip-cidr" {
  default = ["10.0.3.0/24", "10.0.4.0/24", "10.0.5.0/24"]
}

variable "db-ip-cidr" {
  default = ["10.0.6.0/24", "10.0.7.0/24", "10.0.8.0/24"]
}

variable "nat_server_status" {
  type    = bool
  default = false
}

variable "env" {
  default = "Developement"
}

variable "project" {
  default = "smartminingsolution"
}