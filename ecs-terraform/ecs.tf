resource "aws_ecs_cluster" "smart_mining_solution_com" {
  name = "smart_mining_solution"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}
