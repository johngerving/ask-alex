describe('Test chat page', () => {
  beforeEach(() => {
    cy.visit('http://dgx4.humboldt.edu:8080/chat')
  })
  it('Loads', () => {
    cy.contains("Chat")
    cy.get('#message').should('exist')
  })

  it('Sends a message', () => {
    /* ==== Generated with Cypress Studio ==== */
    cy.get('#message').click();
    cy.get('#message').type('What is the capital of France?')
    cy.get('#submit').click();
    /* ==== End Cypress Studio ==== */
    /* ==== Generated with Cypress Studio ==== */
    cy.get('.justify-end > .flex-col').should('be.visible');
    cy.get('.justify-end > .flex-col > .font-normal').should('have.text', 'What is the capital of France?');
    /* ==== End Cypress Studio ==== */
    cy.get('#messages > :nth-child(2) > .flex-col', {timeout: 10000}).should('be.visible');
    cy.get('#message').should('have.value', '');
  })
})
