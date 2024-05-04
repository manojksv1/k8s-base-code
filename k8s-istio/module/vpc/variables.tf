variable "awsregion" {
  default     = "ap-south-1"
  description = "AWS region"
}

# data "aws_availability_zones" "current" {
#   state = "available"
#   filter {
#     name   = "region-name"
#     values = [var.awsregion]
#   }
# }

# variable "availability_zone_list" {
#   description = "az list"
#   type        = list(string)
#   default     = data.aws_availability_zones.current.names
# }


variable "availability_zone_list" {
  description = "az list"
  type        = list(string)
  default     = ["ap-south-1a", "ap-south-1b", "ap-south-1c"]
}

variable "vpc-cidr" {
  default = "10.0.0.0/16"
}

variable "public-cidr" {
  type        = list(string)
  description = "public subnet cidr"
  default     = ["10.0.0.0/24", "10.0.1.0/24", "10.0.2.0/24"]
}

variable "private-cidr" {
  type        = list(string)
  description = "private subnet cidr"
  default     = ["10.0.3.0/24", "10.0.4.0/24", "10.0.5.0/24"]

}

variable "db-cidr" {
  default = ["10.0.6.0/24", "10.0.7.0/24", "10.0.8.0/24"]
}

variable "nat-server-instance-type" {
  description = "nat server instnace"
  default     = "t2.micro"
}

variable "nat_server_status" {
  description = "If you want to add nat server or not"
  type = bool
  default = false
}