variable "cluster-name" {
  default = "test-cluster"
}

variable "ecs_service_role" {
}

variable "lb_target" {
}

variable "container-name" {
  default = "web-page"
}

variable "aws_region" {
  
}
variable "container-port" {
  default = 80
}

variable "repo-name" {
  
}

variable "azlist" {
  
}

variable "ecs_image_ami" {
  type    = string
  default = "ami-072aaf1b030a33b6e"
  # run the following command to get the image ami for your region
  # aws ssm get-parameters --names /aws/service/ecs/optimized-ami/amazon-linux-2/recommended
}

variable "ecs_agent" {
  
}

variable "public_subnet" {
  
}

variable "web_vpc" {
  
}