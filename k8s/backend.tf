terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.44.0"
    }
  }
}

terraform {
  backend "s3" {
    bucket         = "terraformstatefilecopymanoj"
    key            = "terraform/default.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "terraform_lock"
  }
}