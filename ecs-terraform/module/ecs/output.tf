output "ecs-name" {
  value = aws_ecs_cluster.smart_mining_solution_com.name
}

output "ServiceName" {
  value = aws_ecs_service.python_service
}