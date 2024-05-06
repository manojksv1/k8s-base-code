resource "aws_lb_target_group" "web-ui" {
  name        = "web-ui"
  port        = 80
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vic-id
  health_check {
    path                = "/"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 3
    unhealthy_threshold = 3
    matcher             = "200-299"
  }
}

resource "aws_lb_target_group" "api" {
  name        = "api"
  port        = 80
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vic-id
  health_check {
    path                = "/api"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 3
    unhealthy_threshold = 3
    matcher             = "200-299"
  }
}

resource "aws_lb_listener" "smartmining" {
  load_balancer_arn = aws_lb.loadbalancer.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      message_body = "Fixed response content"
      status_code  = "200"
    }
  }
}

resource "aws_lb_listener_rule" "web" {
  listener_arn = aws_lb_listener.smartmining.arn
  priority     = 2

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.web-ui.arn
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }
  tags = {
    name = "web-policy"
  }
}

resource "aws_lb_listener_rule" "api-lis" {
  listener_arn = aws_lb_listener.smartmining.arn
  priority     = 1

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    path_pattern {
      values = ["/api*"]
    }
  }
  tags = {
    name = "api-policy"
  }
}

