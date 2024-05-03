output "combined-subnet-ids" {
  value = concat(module.vpc.private-subnet, module.vpc.public-subnet)
}
