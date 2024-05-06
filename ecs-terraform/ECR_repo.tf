resource "aws_ecr_repository" "python_app_repo" {
  name                 = "smart_mining_solution"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
