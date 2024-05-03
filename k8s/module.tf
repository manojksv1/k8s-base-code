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