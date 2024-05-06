resource "aws_ecs_cluster" "smart_mining_solution_com" {
  name = var.cluster-name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}
