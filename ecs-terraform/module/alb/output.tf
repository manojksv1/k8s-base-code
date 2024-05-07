output "alb-dns" {
  value = aws_lb.loadbalancer.dns_name
}
output "alb_target" {
  value = aws_lb_target_group.web-ui
}