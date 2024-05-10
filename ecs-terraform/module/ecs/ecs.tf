resource "aws_ecs_cluster" "smart_mining_solution_com" {
  name = var.cluster-name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}
# data "aws_caller_identity" "current" {}

# resource "aws_ecs_service" "python_service" {
#   name            = "front-end"
#   cluster         = aws_ecs_cluster.smart_mining_solution_com.id
#   task_definition = aws_ecs_task_definition.python_task.arn
#   desired_count   = 1
#   iam_role        = var.ecs_service_role.arn

#   ordered_placement_strategy {
#     type  = "binpack"
#     field = "cpu"
#   }

#   load_balancer {
#     target_group_arn = var.lb_target.arn
#     container_name   = var.container-name
#     container_port   = var.container-port
#   }

#   # placement_constraints {
#   #   type       = "memberOf"
#   #   expression = "attribute:ecs.availability-zone in [us-west-2a, us-west-2b]"
#   # }
# }

# resource "aws_ecs_task_definition" "python_task" {
#   family = "service"
#   container_definitions = jsonencode([
#     {
#       name      = var.container-name
#       image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.repo-name}:latest"
#       cpu       = 50
#       memory    = 128
#       essential = true
#       portMappings = [
#         {
#           containerPort = var.container-port
#           hostPort      = 0
#         }
#       ]
#     },
#   ])

#   volume {
#     name      = "service-storage"
#     host_path = "/ecs/service-storage"
#   }

#   placement_constraints {
#     type       = "memberOf"
#     expression = "attribute:ecs.availability-zone in [ap-southeast-1a, ap-southeast-1b, ap-southeast-1c]"
#   }
# }


# resource "aws_launch_configuration" "ecs_launch_config" {
#   image_id             = var.ecs_image_ami
#   iam_instance_profile = var.ecs_agent.name
#   security_groups      = [aws_security_group.allow_ecs.id]
#   user_data            = "#!/bin/bash\necho ECS_CLUSTER=${aws_ecs_cluster.smart_mining_solution_com.name} >> /etc/ecs/ecs.config"
#   instance_type        = "t2.medium"
# }

# resource "aws_autoscaling_group" "ecs_auto_scaling_group" {
#   name                 = "asg"
#   vpc_zone_identifier  = [var.public_subnet]
#   launch_configuration = aws_launch_configuration.ecs_launch_config.name

#   desired_capacity          = 1
#   min_size                  = 1
#   max_size                  = 1
#   health_check_grace_period = 300
#   health_check_type         = "EC2"
# }

# resource "aws_security_group" "allow_ecs" {
#   name        = "allow_ecs"
#   description = "Allow all inbound traffic"
#   vpc_id      = var.web_vpc

#   ingress {
#     from_port   = 80
#     to_port     = 80
#     protocol    = "tcp"
#     cidr_blocks = ["0.0.0.0/0"]
#   }

#   egress {
#     from_port   = 0
#     to_port     = 0
#     protocol    = "-1"
#     cidr_blocks = ["0.0.0.0/0"]
#   }


#   tags = {
#     Name = "allow_ecs"
#   }
# }