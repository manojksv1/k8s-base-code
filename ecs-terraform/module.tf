module "avalability-zones" {
  source     = "./module/avalabality-zones"
  aws-region = var.aws-region
}

module "vpc" {
  source                 = "./module/vpc"
  awsregion              = var.aws-region
  availability_zone_list = module.avalability-zones.aws-azs
  public-cidr            = var.public-ip-cidr
  private-cidr           = var.private-ip-cidr
  db-cidr                = var.db-ip-cidr
  nat_server_status      = var.nat_server_status
}

module "ALB" {
  source             = "./module/alb"
  alb_name           = var.alb-name
  vic-id             = module.vpc.vpc
  Environment        = var.env
  public-subnet-list = module.vpc.public-subnet
}

module "ecr" {
  source    = "./module/ecr"
  repo-name = var.Ecr-repo-name
}

module "ecs" {
  source       = "./module/ecs"
  cluster-name = var.cluster-name
}
module "artifacts-bucket" {
  source      = "./module/artifacts_buckets"
  env_name    = var.env
  aws_region  = var.aws-region
  bucket_name = var.artifact-bucket-name
}